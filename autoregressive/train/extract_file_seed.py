# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py

import warnings
warnings.filterwarnings("ignore")
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
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
# from condition.hed import HEDdetector, ControlNetHED_Apache2
import cv2
from torch.nn.parallel import DataParallel
from einops import rearrange
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from language.t5 import T5Embedder
# from condition.canny import CannyDetector
# from condition.hed import HEDdetector
# from condition.lineart import LineArt
#################################################################################
#                                  Training Loop                                #
#################################################################################
import pdb

from datasets import load_from_disk
import io
import numpy as np
import PIL
from PIL import Image
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from tqdm import tqdm
import json

class Seed_Dataset(Dataset):
    def __init__(self,
                 ):
        dataset_folder = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash'
        self.dataset_path = []
        for json_file in tqdm(os.listdir(os.path.join(dataset_folder, 'annotations'))):
            with open(os.path.join(dataset_folder, 'annotations', json_file), 'r') as fr:
                for data in fr:
                    data = data.strip()
                    data = json.loads(data)
                    if data['match_score'] > 0.9:
                        self.dataset_path.append(data)
        for json_file in tqdm(os.listdir(os.path.join(dataset_folder, 'annotations_GPT4V'))):
            with open(os.path.join(dataset_folder, 'annotations_GPT4V', json_file), 'r') as fr:
                for data in fr:
                    data = data.strip()
                    data = json.loads(data)
                    if data['match_score'] > 0.9:
                        self.dataset_path.append(data)
        self.length = len(self.dataset_path)

    def __len__(self,):
        return len(self.dataset_path)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img

    # def collate_fn(self, examples):
    #     input_img = torch.stack([example["input_img"] for example in examples])
    #     edited_img = torch.stack([example["edited_img"] for example in examples])
    #     captions = [example["caption"] for example in examples]
    #     images = [example["image"] for example in examples]
    
    #     return {
    #             'input_imgs': input_img,
    #             'edited_imgs': edited_img,
    #             "captions": captions,
    #             "images": images,
    #             }
    
    
    def __getitem__(self, index):
        # if index < 710000:
        #     return {}
        data = self.dataset_path[index]
        input_img = os.path.join('/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash', data['source_image'])
        edited_img = os.path.join('/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash', data['target_image'])
        input_img = Image.open(input_img).convert('RGB')
        edited_img = Image.open(edited_img).convert('RGB')
        edit_txt = data['instruction']
        
        image = input_img
        input_img_PIL = input_img
        edited_img_PIL = edited_img
        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img)

        return {
                'input_img': input_img,
                'edited_img': edited_img,
                "caption": edit_txt,
                "image": image.resize((512,512)),
                "input_img_PIL": input_img_PIL,
                "edited_img_PIL": edited_img_PIL
                }

def collate_fn(examples):
    input_img = torch.stack([example["input_img"] for example in examples])
    edited_img = torch.stack([example["edited_img"] for example in examples])
    captions = [example["caption"] for example in examples]
    images = [example["image"] for example in examples]
    input_img_PIL = [example["input_img_PIL"] for example in examples]
    edited_img_PIL = [example["edited_img_PIL"] for example in examples]

    return {
            'input_imgs': input_img,
            'edited_imgs': edited_img,
            "captions": captions,
            "images": images,
            "input_img_PIL": input_img_PIL,
            "edited_img_PIL": edited_img_PIL
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
        # os.makedirs(os.path.join(args.code_path, f'code'), exist_ok=True)

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
    qwenvl_weight = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/Qwen3-VL-2B-Instruct'
    processor = AutoProcessor.from_pretrained(qwenvl_weight)
    qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
    qwenvl_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    qwenvl_model.to(device)

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
    dataset = Seed_Dataset()
    # dataset = Subset(dataset, range(714556, len(dataset)))
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
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
    idx = 0
    for batch in tqdm(loader):
        idx += 1
        if idx > 100000:
            break
        captions = batch['captions']
        input_imgs = batch['input_imgs']
        edited_imgs = batch['edited_imgs']
        images = batch['images']
        train_steps = rank + total + 713485
        # if train_steps < 713500:
        #     pass
        # else:
        if not os.path.exists(os.path.join(args.code_path, str(train_steps // 10000))):
            os.makedirs(os.path.join(args.code_path, str(train_steps // 10000)), exist_ok=True)
            os.makedirs(os.path.join('/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash/input_img', str(train_steps // 10000)), exist_ok=True)
            os.makedirs(os.path.join('/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash/edited_img', str(train_steps // 10000)), exist_ok=True)

        with torch.no_grad():
            _, _, [_, _, input_indices] = vq_model.encode(batch['input_imgs'].to(device))
            _, _, [_, _, edited_indices] = vq_model.encode(batch['edited_imgs'].to(device))
            caption_emb, emb_mask = t5_model.get_text_embeddings(captions)
            valid_num = int(emb_mask.sum().item())
            caption_emb = caption_emb[:, :valid_num]
            
            prompts = []
            for caption, image in zip(captions, images):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {
                        "type": "image",
                        },  
                    ],
                }]
                prompts.append(processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ))
            qwenvl_inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            qwenvl_inputs = qwenvl_inputs.to(device)
            qwenvl_outputs = qwenvl_model.forward(
                **qwenvl_inputs,
                output_hidden_states=True
            )
            qwenvl_embedding = qwenvl_outputs['hidden_states'][-1][0]#, -256:]
            
        
        input_code = input_indices.reshape(1, 1, -1)
        edited_code = edited_indices.reshape(1, 1, -1)
        input_code = input_code.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        edited_code = edited_code.detach().cpu().numpy()
        caption_emb = caption_emb.to(torch.float16).detach().cpu().numpy()
        input_img_PIL = batch['input_img_PIL'][0]#(255*(batch['input_imgs'][0].permute(1,2,0) * 0.5 + 0.5)).cpu().detach().numpy().astype(np.uint8)
        edited_img_PIL = batch['edited_img_PIL'][0]#(255*(batch['edited_imgs'][0].permute(1,2,0) * 0.5 + 0.5)).cpu().detach().numpy().astype(np.uint8)
        # import pdb;pdb.set_trace()
        qwenvl_embedding = qwenvl_embedding.to(torch.float16).cpu().detach().numpy()
        save_dict = {}
        save_dict['prompt'] = captions
        save_dict['caption_emb'] = caption_emb
        save_dict['input_code'] = input_code
        save_dict['edited_code'] = edited_code
        # save_dict['input_img_bytes'] = input_img_bytes
        # save_dict['edited_img_bytes'] = edited_img_bytes
        save_dict['qwenvl_embedding'] = qwenvl_embedding
        input_img_PIL.save(f'/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash/input_img/{train_steps // 10000}/{train_steps}.png')
        edited_img_PIL.save(f'/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash/edited_img/{train_steps // 10000}/{train_steps}.png')
        np.savez(f'{args.code_path}/{train_steps // 10000}/{train_steps}.npz', **save_dict)
            # np.savez(f'/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/autoregressive/train/{train_steps}.npz', **save_dict)
            # print(save_dict['qwenvl_embedding'].shape, f'{args.code_path}/{train_steps // 10000}/{train_steps}.npz')
            # torch.save(save_dict, f'{args.code_path}/{train_steps // 10000}/{train_steps}.pt', _use_new_zipfile_serialization=True)
        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        # break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, default="/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash/code")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, help="ckpt path for vq model", default="/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt")
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
    parser.add_argument("--t5-path", type=str, default='/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()
    main(args)
