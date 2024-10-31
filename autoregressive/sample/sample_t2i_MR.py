import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.gpt_t2i import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataset.t2i_control import build_t2i_control_code
from accelerate import Accelerator
from dataset.build import build_dataset
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
from condition.canny import CannyDetector
from condition.hed import HEDdetector
import numpy as np
from PIL import Image
from condition.lineart import LineArt
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
from condition.midas.depth import MidasDetector


def resize_image_to_16_multiple(image_path, condition_type='seg'):
    image = Image.open(image_path)
    width, height = image.size
    
    if condition_type == 'depth':  # The depth model requires a side length that is a multiple of 32
        new_width = (width + 31) // 32 * 32
        new_height = (height + 31) // 32 * 32
    else:
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16

    resized_image = image.resize((new_width, new_height))
    return resized_image

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        condition_type=args.condition_type,
    ).to(device=device, dtype=precision)

    _, file_extension = os.path.splitext(args.gpt_ckpt)
    if file_extension.lower() == '.safetensors':
        from safetensors.torch import load_file
        model_weight = load_file(args.gpt_ckpt)
        gpt_model.load_state_dict(model_weight, strict=False)
        gpt_model.eval()
    else:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        if "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint: # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        gpt_model.load_state_dict(model_weight, strict=False)
        gpt_model.eval()
        del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    

    if args.condition_type == 'canny':
        get_control = CannyDetector()
    elif args.condition_type == 'hed':
        get_control = HEDdetector().to(device).eval()
    elif args.condition_type == 'lineart':
        get_control = LineArt()
        get_control.load_state_dict(torch.load('condition/ckpts/model.pth', map_location=torch.device('cpu')))
        get_control.to(device)
    elif args.condition_type == 'depth':
        processor = DPTImageProcessor.from_pretrained("condition/ckpts/dpt_large")
        model_large = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large").to(device)
        model = MidasDetector(device=device)
    with torch.no_grad():
        
        condition_img = resize_image_to_16_multiple(args.condition_path, args.condition_type)
        W, H = condition_img.size
        print(H,W)
        if args.condition_type == 'seg':
            condition_img = torch.from_numpy(np.array(condition_img))
            condition_img = condition_img.permute(2,0,1).unsqueeze(0).repeat(2,1,1,1)
        elif args.condition_type == 'canny':
            condition_img = get_control(np.array(condition_img))
            condition_img = torch.from_numpy(condition_img[None,None,...]).repeat(2,3,1,1)
        elif args.condition_type == 'hed':
            condition_img = get_control(torch.from_numpy(np.array(condition_img)).permute(2,0,1).unsqueeze(0).to(device))
            condition_img = condition_img.unsqueeze(1).repeat(2,3,1,1)
        elif args.condition_type == 'lineart':
            condition_img = get_control(torch.from_numpy(np.array(condition_img)).permute(2,0,1).unsqueeze(0).to(device).float())
            condition_img = condition_img.repeat(2,3,1,1) * 255
        elif args.condition_type == 'depth':
            images = condition_img
            if H == W:
                inputs = processor(images=images, return_tensors="pt", size=(H,W)).to(device)
                outputs = model_large(**inputs)
                condition_img = outputs.predicted_depth
                condition_img = (condition_img * 255 / condition_img.max())
            else:
                condition_img = torch.from_numpy(model(torch.from_numpy(np.array(condition_img)).to(device))).unsqueeze(0)
            condition_img = condition_img.unsqueeze(0).repeat(2,3,1,1)
        condition_img = condition_img.to(device)
        condition_img = 2*(condition_img/255 - 0.5)
        prompts = [args.prompt if args.prompt is not None else "a high-quality image"]
        prompts = prompts * 2
        caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

        if not args.no_left_padding:
            print(f"processing left-padding...")    
            # a naive way to implement left-padding
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
                valid_num = int(emb_mask.sum().item())
                print(f'  prompt {idx} token len: {valid_num}')
                new_caption_emb = torch.cat([caption_emb[valid_num:],caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks
        qzshape = [len(c_indices), args.codebook_embed_dim, H//args.downsample_size, W//args.downsample_size]
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, (H//args.downsample_size)*(W//args.downsample_size),#latent_size ** 2, 
            c_emb_masks, condition=condition_img.to(precision),
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )
        sampling_time = time.time() - t1
        print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
        
        t2 = time.time()
        print(index_sample.shape)
        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        decoder_time = time.time() - t2
        print(f"decoder takes about {decoder_time:.2f} seconds.")

        samples = torch.cat((condition_img[0:1], samples), dim=0)
        save_image(samples, f"sample/example/sample_t2i_MR_{args.condition_type}.png", nrow=4, normalize=True, value_range=(-1, 1))
        print(f"image is saved to sample/example/sample_t2i_MR_{args.condition_type}.png")
        print(prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='checkpoints/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768], default=768)
    parser.add_argument("--image-H", type=int, default=512)
    parser.add_argument("--image-W", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--cfg-scale", type=float, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--condition-type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth'], default="canny")
    parser.add_argument("--prompt", type=str, default='a high-quality image')
    parser.add_argument("--condition-path", type=str, default='condition/example/t2i/multigen/landscape.png')
    args = parser.parse_args()
    main(args)
