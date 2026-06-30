# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import os
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)

from PIL import Image
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from functools import partial
import torch.nn.functional as F
import numpy as np
import cv2


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        condition_token_num=args.condition_token_nums,
        image_size=args.image_size
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

    condition_null = None
    if args.condition_type == 'canny':
        sample_list = [650, 2312, 15000, 48850]  # canny
    elif args.condition_type == 'depth':
        sample_list = [101, 4351, 10601, 48901]

    class_labels = [np.load(f"condition/example/c2i/{args.condition_type}/{i}.npy")[0] for i in sample_list]
    condition_imgs = [np.array(Image.open((f"condition/example/c2i/{args.condition_type}/{i}.png")))[None,None,...] for i in sample_list]
    condition_imgs = torch.from_numpy(np.concatenate(condition_imgs, axis=0)).to(device).to(torch.float32)/255
    condition_imgs = 2*(condition_imgs-0.5)
    print(condition_imgs.shape)
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
    t1 = time.time()

    index_sample = generate(
        gpt_model, c_indices, latent_size ** 2, condition=condition_imgs.repeat(1,3,1,1).to(precision), condition_null=condition_null, condition_token_nums=args.condition_token_nums,
        cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, sample_logits=True, 
        )

    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")    
    
    t2 = time.time()
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    decoder_time = time.time() - t2
    print(f"decoder takes about {decoder_time:.2f} seconds.")
    # Save and display images:
    condition_imgs = condition_imgs.repeat(1,3,1,1)
    samples = torch.cat((condition_imgs[:4], samples[:4]),dim=0)
    save_image(samples, f"sample/example/sample_{args.gpt_type}_{args.condition_type}.png", nrow=4, normalize=True, value_range=(-1, 1))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--condition-token-nums", type=int, default=0)
    parser.add_argument("--condition-type", type=str, default='canny', choices=['canny', 'depth'])
    args = parser.parse_args()
    main(args)