# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import warnings
warnings.filterwarnings('ignore')
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
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt_t2i import GPT_models
from autoregressive.models.generate import generate
from condition.hed import HEDdetector, nms
from condition.canny import CannyDetector
from autoregressive.test.metric import SSIM, F1score, RMSE
from condition.midas.depth import MidasDetector
import torch.distributed as dist
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm
from functools import partial
from dataset.t2i_control import build_t2i_control_code
from language.t5 import T5Embedder
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from condition.lineart import LineArt
import torch.nn.functional as F

def main(args):
    # # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
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
        adapter_size=args.adapter_size,
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
    

    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = build_t2i_control_code(args)
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )    


    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        pass
        # print(f"no need to compile model in demo") 
    
    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    # import pdb;pdb.set_trace()
    date = os.path.split(os.path.dirname(os.path.dirname(os.path.dirname(args.gpt_ckpt))))[-1]
    folder_name = f"{model_string_name}-{date}-{ckpt_string_name}-size-{args.image_size}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}"
    if rank == 0:
        if args.save_image:
            os.makedirs(sample_folder_dir, exist_ok=True)
            os.makedirs(f"{args.sample_dir}/visualization", exist_ok=True)
            os.makedirs(f"{args.sample_dir}/annotations", exist_ok=True)
            print(f"Saving .png samples at {sample_folder_dir}")
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total = 0
    
    if args.condition_type == 'hed':
        get_condition = HEDdetector().to(device).eval()
    elif args.condition_type == 'canny':
        get_condition = CannyDetector()
    elif args.condition_type == 'lineart':
        get_condition = LineArt()
        get_condition.load_state_dict(torch.load('condition/ckpts/model.pth', map_location=torch.device('cpu')))
        get_condition.to(device)

    condition_null = None
    num = 0
    print(len(loader))
    for batch in tqdm(loader):
        num += 1
        # if num>2:
        #     break
        prompts = batch['prompt']
        condition_imgs = batch['control'].to(device)
        
        if args.condition_type in ['hed', 'lineart']:
            with torch.no_grad():
                condition_imgs = get_condition(condition_imgs.float())
                if args.condition_type == 'hed':
                    condition_imgs = condition_imgs.unsqueeze(1)/255
                # if args.condition_type == 'lineart':
                #     condition_imgs = 1 - condition_imgs
                condition_imgs = condition_imgs.repeat(1,3,1,1)
                condition_imgs = 2*(condition_imgs - 0.5)
        # condition_origin = condition_imgs.clone()

        if args.condition_type == 'seg':
            labels = batch['label']

        
        caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)

        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            new_caption_emb = torch.cat([caption_emb[valid_num:],caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks

        qzshape = [len(c_indices), args.codebook_embed_dim, args.image_H//args.downsample_size, args.image_W//args.downsample_size]

        index_sample = generate(
            gpt_model, c_indices, (args.image_H//args.downsample_size)*(args.image_W//args.downsample_size), c_emb_masks, condition=condition_imgs.to(precision),
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )  

        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]

        for i in range(len(samples)):
            # # Save samples to disk as individual .png files
            index = i * dist.get_world_size() + rank + total
            if args.save_image:
                save_image(samples[i], f"{args.sample_dir}/visualization/{index:06d}.png", nrow=1, normalize=True, value_range=(-1, 1))
                save_image(condition_imgs[i,0], f"{args.sample_dir}/annotations/{index:06d}.png", nrow=1, normalize=True, value_range=(-1, 1))
            if args.condition_type == 'seg':
                Image.fromarray(labels[i].numpy().astype('uint8'), mode='L').save(f"{args.sample_dir}/annotations/{index:06d}.png")
        total += global_batch_size



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512, 768], default=512)
    parser.add_argument("--image-H", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768, 832, 960, 1024], default=512)
    parser.add_argument("--image-W", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768, 832, 960, 1024], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000,help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--condition", type=str, default='hed', choices=['canny', 'hed'])
    parser.add_argument("--per-proc-batch-size", type=int, default=25)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco', 'imagenet_code'], default='imagenet_code')
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=2000)
    parser.add_argument("--save-image", type=bool, default=True)
    parser.add_argument("--t5-path", type=str, default='checkpoints/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--code-path", type=str, default="code")
    parser.add_argument("--code-path2", type=str, default=None)
    parser.add_argument("--get-image", type=bool, default=False)
    parser.add_argument("--get-prompt", type=bool, default=True)
    parser.add_argument("--get-label", type=bool, default=False)
    parser.add_argument("--condition-type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth'], default="canny")
    parser.add_argument("--adapter-size", type=str, default="small")
    args = parser.parse_args()
    main(args)