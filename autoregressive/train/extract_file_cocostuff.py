# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)
from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models
from condition.hed import HEDdetector, ControlNetHED_Apache2
import cv2
from torch.nn.parallel import DataParallel
from einops import rearrange
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from language.t5 import T5Embedder
#################################################################################
#                                  Training Loop                                #
#################################################################################
resolution = (512, 512)
image_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
        ]
    )
label_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
        ]
    )

def collate_fn(examples):
    
    pil_images = [example['image'].convert("RGB") for example in examples]
    images = [image_transforms(image) for image in pil_images]
    images = torch.stack(images)
    
    conditioning_images = [example['control_seg'].convert("RGB") for example in examples]
    conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]
    conditioning_images = torch.stack(conditioning_images)
    
    captions = [example['prompt'] for example in examples]
    
    dtype = torch.long
    # labels = [torch.from_numpy(np.array(example['panoptic_seg_map'])).unsqueeze(0) for example in examples]  # seg_map  panoptic_seg_map
    # labels = [label_image_transforms(label) for label in labels]
    # labels = torch.stack(labels)
    labels = [example['panoptic_seg_map'] for example in examples]
    

    return {
        "images": images,  # -1~1
        "conditioning_images": conditioning_images,  # 0~1
        "captions": captions,
        "labels": labels
    }

def main(args):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup a feature folder:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'code'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'image'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'control'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'caption_emb'), exist_ok=True)
        if args.split == 'validation':
            os.makedirs(os.path.join(args.code_path, f'label'), exist_ok=True)
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        model_max_length=args.t5_feature_max_len,
    )

    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = load_dataset(
                    args.data_path,
                    cache_dir=None,
                )
    if not args.debug:
        sampler = DistributedSampler(
            dataset[args.split],
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset[args.split],
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    from tqdm import tqdm
    total = 0
    code_len = 1024
    t5_feature_max_len = 120
    t5_feature_dim = 2048
    max_seq_length = t5_feature_max_len + code_len
    for batch in tqdm(loader):
        
        captions = batch['captions']
        
        train_steps = rank + total
        img_save_path = f'{args.code_path}/image/{train_steps}.png'
        cond_save_path = f'{args.code_path}/control/{train_steps}.png'
        label_save_path = f'{args.code_path}/label/{train_steps}.png'
        Image.fromarray((255*(batch['images'][0].numpy().transpose(1,2,0)*0.5+0.5)).astype('uint8'), mode='RGB').save(img_save_path)
        Image.fromarray((255*batch['conditioning_images'][0].numpy().transpose(1,2,0)).astype('uint8'), mode='RGB').save(cond_save_path)
        
        label = Image.fromarray(np.array(batch['labels'][0]).astype('uint8'))
        label.resize((512,512), Image.Resampling.NEAREST).save(label_save_path)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(batch['images'].to(device))
            
            caption_emb, emb_mask = t5_model.get_text_embeddings(captions)
            valid_num = int(emb_mask.sum().item())
            caption_emb = caption_emb[:, :valid_num]
        
        codes = indices.reshape(1, 1, -1)
        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        np.save(f'{args.code_path}/code/{train_steps}.npy', x)

        caption_emb = caption_emb.to(torch.float32).detach().cpu().numpy()
        caption_dict = {}
        caption_dict['prompt'] = captions
        caption_dict['caption_emb'] = caption_emb
        np.savez(f'{args.code_path}/caption_emb/{train_steps}.npz', **caption_dict)
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--min-threshold", type=int, default=200)
    parser.add_argument("--max-threshold", type=int, default=400)
    parser.add_argument("--t5-path", type=str, default='checkpoints/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()
    main(args)
