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
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from condition.hed import HEDdetector, nms
from condition.canny import CannyDetector
from condition.midas.depth import MidasDetector
from autoregressive.test.metric import SSIM, F1score, RMSE
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
from skimage.transform import resize
from torch.nn.functional import interpolate
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
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
        condition_token_num=args.condition_token_nums,
        # image_size=args.image_size
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
    
    if args.condition_type == 'hed':
        get_condition = HEDdetector(device=device)
        get_metric = SSIM()
    elif args.condition_type == 'canny':
        get_condition = CannyDetector()
        get_metric = F1score()
    elif args.condition_type == 'depth':
        get_condition = MidasDetector(device=device)
        get_metric = RMSE()
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.dataset == 'imagenet':
        dataset = build_dataset(args, transform=transform)
    elif args.dataset == 'coco':
        dataset = build_dataset(args, transform=transform)
    elif args.dataset == 'imagenet_code':
        dataset = build_dataset(args)
    else:
        raise Exception("please check dataset")
    
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
        drop_last=False
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

    date = os.path.split(os.path.dirname(os.path.dirname(os.path.dirname(args.gpt_ckpt))))[-1]

    sample_folder_dir = f"{args.sample_dir}/imagenet/{args.condition_type}"
    if rank == 0:
        if args.save_image:
            os.makedirs(sample_folder_dir, exist_ok=True)
            print(f"Saving .png samples at {sample_folder_dir}")
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total = 0

    condition_null = None
    num = 0
    for batch in tqdm(loader):
        num += 1
        # if num > 40:
        #     break
        class_labels = batch["labels"].to(device).squeeze(1)
        condition_image = batch["condition_img"].to(device)
        condition_imgs = batch["condition_imgs"].to(device)

        batch_size = class_labels.shape[0]
        
        c_indices = class_labels
        qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]

        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, condition=condition_imgs.repeat(1,3,1,1).to(precision), condition_null=condition_null, condition_token_nums=args.condition_token_nums,
            cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True, 
            )

        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        samples = 255*(samples*0.5 + 0.5)
        if samples.shape[2] != 256:
            samples = interpolate(samples, size=(256, 256), mode='bilinear', align_corners=False)
       
        condition_imgs = 255*(condition_imgs*0.5 + 0.5)
        if condition_imgs.shape[2] != 256:
            condition_imgs = interpolate(condition_imgs, size=(256, 256), mode='bilinear', align_corners=False)
        for i in range(len(samples)):
           
            sample = samples[i].to(torch.uint8).permute(1,2,0)
            sample_condition = get_condition(sample)
            if torch.is_tensor(sample_condition):
                sample_condition = sample_condition.cpu().numpy()
            condition_img = condition_imgs[i,0].cpu().detach().numpy()

            get_metric.update(condition_img, sample_condition)

            index = i * dist.get_world_size() + rank + total
            if args.save_image:
                save_image(2*(samples[i]/255 - 0.5), f"{sample_folder_dir}/{index:06d}.png", nrow=1, normalize=True, value_range=(-1, 1))

        total += global_batch_size
    
    metric = get_metric.calculate()
    print(f'count: {get_metric.count}')
    print(f'{args.condition_type}: {metric}')


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
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--get-condition-img", type=bool, default=False)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco', 'imagenet_code'], default='imagenet_code')
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--save-image", type=bool, default=False)
    args = parser.parse_args()
    main(args)