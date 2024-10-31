from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
import torch
from datasets import load_dataset, load_from_disk
import random
import pickle
import logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk, concatenate_datasets
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer, PretrainedConfig
import argparse
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from packaging import version
from torchvision import transforms
from torch.cuda.amp import autocast
from torchvision.transforms.functional import normalize

from dataset.utils import group_random_crop
import numpy as np
import os
from language.t5 import T5Embedder
from torch.utils.data import Dataset
from condition.canny import CannyDetector
# from condition.hed import HEDdetector


logger = get_logger(__name__)

class T2IControlCode(Dataset):
    def __init__(self, args):
        self.get_image = args.get_image
        self.get_prompt = args.get_prompt
        self.get_label = args.get_label
        self.control_type = args.condition_type
        if self.control_type == 'canny':
            self.get_control = CannyDetector()
        
        self.code_path = args.code_path
        code_file_path = os.path.join(self.code_path, 'code')
        file_num = len(os.listdir(code_file_path))
        self.code_files = [os.path.join(code_file_path, f"{i}.npy") for i in range(file_num)]
        
        if args.code_path2 is not None:
            self.code_path2 = args.code_path2
            code_file_path2 = os.path.join(self.code_path2, 'code')
            file_num2 = len(os.listdir(code_file_path2))
            self.code_files2 = [os.path.join(code_file_path2, f"{i}.npy") for i in range(file_num2)]
            self.code_files = self.code_files + self.code_files2

        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.code_files)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def collate_fn(self, examples):
        
        code = torch.stack([example["code"] for example in examples])
        control =  torch.stack([example["control"] for example in examples])
        if self.control_type == 'canny':
            control = control.unsqueeze(1).repeat(1,3,1,1)
        caption_emb =  torch.stack([example["caption_emb"] for example in examples])
        attn_mask = torch.stack([example["attn_mask"] for example in examples])
        valid = torch.stack([example["valid"] for example in examples])
        if self.get_image:
            image = torch.stack([example["image"] for example in examples])
        if self.get_prompt:
            prompt = [example["prompt"][0] for example in examples]
        if self.control_type == "seg":
            label = torch.stack([example["label"] for example in examples])
            
        output = {}
        output['code'] = code
        output['control'] = control
        output['caption_emb'] = caption_emb
        output['attn_mask'] = attn_mask
        output['valid'] = valid
        if self.get_image:
            output['image'] = image
        if self.get_prompt:
            output['prompt'] = prompt
        if self.control_type == "seg":
            output['label'] = label
        return output

    def __getitem__(self, index):
        
        
        code_path = self.code_files[index]
        if self.control_type == 'seg':
            control_path = code_path.replace('code', 'control').replace('npy', 'png') 
            control = np.array(Image.open(control_path))/255
            control = 2*(control - 0.5)
        elif self.control_type == 'depth':
            control_path = code_path.replace('code', 'control_depth').replace('npy', 'png')
            control = np.array(Image.open(control_path))/255
            control = 2*(control - 0.5)
        caption_path = code_path.replace('code', 'caption_emb').replace('npy', 'npz') 
        image_path = code_path.replace('code', 'image').replace('npy', 'png')
        label_path = code_path.replace('code', 'label').replace('npy', 'png') 
        
        code = np.load(code_path)
        image = np.array(Image.open(image_path))
        
        
        
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        caption = np.load(caption_path)
        t5_feat = torch.from_numpy(caption['caption_emb'])
        prompt = caption['prompt']
        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
        valid = 1
        
        output = {}
        output['code'] = torch.from_numpy(code)
        if self.control_type == 'canny':
            output['control'] = torch.from_numpy(2*(self.get_control(image)/255 - 0.5))
        elif self.control_type == "seg":
            output['control'] = torch.from_numpy(control.transpose(2,0,1))
        elif self.control_type == "depth":
            output['control'] = torch.from_numpy(control.transpose(2,0,1))
        elif self.control_type == 'hed':
            output['control'] = torch.from_numpy(image.transpose(2,0,1))
        elif self.control_type == 'lineart':
            output['control'] = torch.from_numpy(image.transpose(2,0,1))
        output['caption_emb'] = t5_feat_padding
        output['attn_mask'] = attn_mask
        output['valid'] = torch.tensor(valid)
        output['image'] = torch.from_numpy(image.transpose(2,0,1))
        if self.get_prompt:
            output['prompt'] = prompt
        if self.control_type == "seg":
            output['label'] = torch.from_numpy(np.array(Image.open(label_path)))
        return output


def build_t2i_control_code(args):
    dataset = T2IControlCode(args)
    return dataset
if __name__ == '__main__':

    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    train_dataset, val_dataset = make_train_dataset(args, None, accelerator)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=8,
        num_workers=0,
    )

    from tqdm import tqdm 
    for step, batch in tqdm(enumerate(train_dataloader)):
        continue