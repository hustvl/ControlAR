'''
https://github.com/JitengMu/EditAR/blob/main/dataset/Edit_PIPE.py
'''

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
import os
import torchvision.transforms as T
# PIPE dataset
class PIPE_Dataset(Dataset):
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        self.use_code = args.use_code
        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = args.t5_feature_max_len
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len
        # PIPE Dataset path
        if self.use_code:
            self.dataset_path = []
            for folder in os.listdir(os.path.join(dataset_path, 'code')):
                for file in os.listdir(os.path.join(dataset_path, 'code', folder)):
                    self.dataset_path.append(os.path.join(dataset_path, 'code', folder, file))
        else:
            self.dataset_path = load_from_disk(dataset_path)
            
        length = len(self.dataset_path)
        self.dataset_path = self.dataset_path * 2
        self.dataset_path = self.dataset_path[:int(1.5*length)]
        
        self.qwenvl_image_size = args.qwenvl_image_size
        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.erase = T.Compose([
            T.ToTensor(),
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5)
        ])
        

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

    def collate_fn(self, examples):
        images = [example["image"] for example in examples]
        index = [example["index"] for example in examples]
        mode = [example["mode"] for example in examples]
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_ids_attn_mask = torch.stack([example["input_ids_attn_mask"] for example in examples])
        input_img = torch.stack([example["input_img"] for example in examples])
        edited_img = torch.stack([example["edited_img"] for example in examples])
        attn_mask = torch.stack([example["attn_mask"] for example in examples])
        feat_len = [example["feat_len"] for example in examples]
        emb_mask = torch.stack([example["emb_mask"] for example in examples])
        captions = [example["caption"] for example in examples]
        if self.use_code:
            qwenvl_embedding = torch.stack([example["qwenvl_embedding"] for example in examples])
            caption_embed = torch.stack([example["caption_embed"] for example in examples])
            return {
                    'index': index,
                    'mode': mode,
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'edited_img': edited_img,
                    "attn_mask": attn_mask,
                    "feat_len": feat_len,
                    "emb_mask": emb_mask,
                    "captions": captions,
                    "images": images,
                    "qwenvl_embedding": qwenvl_embedding,
                    "caption_embed": caption_embed,
                    }
        else:
            return {
                    'index': index,
                    'mode': mode,
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'edited_img': edited_img,
                    "attn_mask": attn_mask,
                    "feat_len": feat_len,
                    "emb_mask": emb_mask,
                    "captions": captions,
                    "images": images,
                    }
    
    
    def __getitem__(self, index):
        # IMG_START = self.llm_tokenizer.vocab.boi_id
        # IMG_END = self.llm_tokenizer.vocab.eoi_id
        # Loading Path...
        # data = self.dataset_path[index]
        # ['source_img', 'target_img', 'Instruction_VLM-LLM', 'Instruction_Class', 'Instruction_Ref_Dataset', 'object_location', 'target_img_dataset', 'img_id', 'ann_id']
        if self.use_code:
            try:
                data_path = self.dataset_path[index]
                data = np.load(data_path)
                input_img = os.path.join(data_path.replace('code', 'input_img').replace('.npz', '.png'))
                # if os.path.exists(data_path.replace('code', 'flux_img').replace('.npz', '.png')):
                #     edited_img = os.path.join(data_path.replace('code', 'flux_img').replace('.npz', '.png'))
                # else:
                edited_img = os.path.join(data_path.replace('code', 'edited_img').replace('.npz', '.png'))
                input_img = Image.open(input_img).convert('RGB').resize((self.image_size,self.image_size))
                # input_img = Image.fromarray((self.erase(input_img).permute(1,2,0).numpy()*255).astype(np.uint8))
                edited_img = Image.open(edited_img).convert('RGB').resize((self.image_size,self.image_size))
                edit_txt = data['prompt'].item()
                edited_code = torch.from_numpy(data['edited_code'])
                input_code = torch.from_numpy(data['input_code'])
                qwenvl_embedding = torch.from_numpy(data['qwenvl_embedding'])
            except Exception as e:
                print(f"错误样本：{data_path}")
                index = 0
                data_path = self.dataset_path[index]
                data = np.load(data_path)
                input_img = os.path.join(data_path.replace('code', 'input_img').replace('.npz', '.png'))
                edited_img = os.path.join(data_path.replace('code', 'edited_img').replace('.npz', '.png'))
                input_img = Image.open(input_img).convert('RGB').resize((self.image_size,self.image_size))
                # input_img = Image.fromarray((self.erase(input_img).permute(1,2,0).numpy()*255).astype(np.uint8))
                edited_img = Image.open(edited_img).convert('RGB').resize((self.image_size,self.image_size))
                edit_txt = data['prompt'].item()
                edited_code = torch.from_numpy(data['edited_code'])
                input_code = torch.from_numpy(data['input_code'])
                qwenvl_embedding = torch.from_numpy(data['qwenvl_embedding'])
            
            
            
            qwenvl_embedding = qwenvl_embedding[-256:]
            
            
            
            padding_mask = torch.zeros(306, dtype=torch.bool)
     
            
            
            caption_embed = torch.from_numpy(data['caption_emb'])
            t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
            t5_feat = caption_embed
            t5_feat_len = t5_feat.shape[1] 
            feat_len = min(self.t5_feature_max_len, t5_feat_len)
            t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
            
            # i = 2
            # while i * feat_len < self.t5_feature_max_len:
            #     t5_feat_padding[:, -i*feat_len:-(i-1)*feat_len] = t5_feat[:, :feat_len]
            #     i += 1
            # if i * feat_len == self.t5_feature_max_len:
            #     t5_feat_padding[:, -i*feat_len:-(i-1)*feat_len] = t5_feat[:, :feat_len]
            # else:
            #     t5_feat_padding[:, :self.t5_feature_max_len-(i-1)*feat_len] = t5_feat[:, :self.t5_feature_max_len-(i-1)*feat_len]
            
            
            emb_mask = torch.zeros((self.t5_feature_max_len,))
            emb_mask[-feat_len:] = 1
            attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
            T = self.t5_feature_max_len
            attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
            eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
            attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
            attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
            image = input_img
        else:
            data = self.dataset_path[index]
            # convert into torch style
            if random.random() > 0.5:
                input_img = data['source_img']
                edited_img = data['target_img']
                input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
                edited_img = Image.open(io.BytesIO(edited_img['bytes'])).convert('RGB')
                # edit_txt = data['Instruction_VLM-LLM'] + data['object_location']
                edit_txt = data['Instruction_VLM-LLM']

            else:
                input_img = data['target_img']
                edited_img = data['source_img']
                input_img = Image.open(io.BytesIO(input_img['bytes'])).convert('RGB')
                edited_img = Image.open(io.BytesIO(edited_img['bytes'])).convert('RGB')
                # edit_txt = 'Remove ' + data['Instruction_VLM-LLM'][4:] + data['object_location']
                # edit_txt = 'Remove ' + data['Instruction_VLM-LLM'][4:]
                if data['Instruction_VLM-LLM'][:6] == "Add a ":
                    edit_txt = 'Remove the' + data['Instruction_VLM-LLM'][5:]
                else:
                    edit_txt = 'Remove ' + data['Instruction_VLM-LLM'][4:]
            image = input_img
            edit_text_tokens_and_mask = self.llm_tokenizer(
                edit_txt,
                max_length=self.t5_feature_max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            edit_txt_token = edit_text_tokens_and_mask['input_ids']
            edit_txt_attn_mask = edit_text_tokens_and_mask['attention_mask']
            feat_len = int(edit_text_tokens_and_mask['attention_mask'].sum().item())
            input_ids = edit_txt_token[0]
            input_ids_attn_mask = edit_txt_attn_mask[0]
            
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)

        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img, target_image_size=self.image_size)
        edited_img = self._whiten_transparency(edited_img)
        edited_img = self._vqgan_input_from(edited_img, target_image_size=self.image_size)

      

        if self.use_code:
            return {
                'index': index,
                'dataset': 'pipe',
                'mode': 1,
                # 'input_ids': input_ids,
                # 'input_ids_attn_mask': input_ids_attn_mask,
                'input_img': input_img,
                'edited_img': edited_img,
                "attn_mask": attn_mask,
                "caption": edit_txt,
                "image": image.resize((self.qwenvl_image_size, self.qwenvl_image_size), resample=Image.Resampling.LANCZOS),
                'qwenvl_embedding': qwenvl_embedding,
                "padding_mask": padding_mask,
                'caption_embed': t5_feat_padding.squeeze(0),
                'input_code': input_code.squeeze(0).squeeze(0),
                'edited_code': edited_code.squeeze(0).squeeze(0),
                }
        else:
            return {
                    'index': index,
                    'dataset': 'pipe',
                    'mode': 1,
                    'input_ids': input_ids,
                    'input_ids_attn_mask': input_ids_attn_mask,
                    'input_img': input_img,
                    'edited_img': edited_img,
                    "attn_mask": attn_mask,
                    "feat_len": feat_len,
                    "emb_mask": emb_mask,
                    "caption": edit_txt,
                    "image": image.resize((self.qwenvl_image_size, self.qwenvl_image_size), resample=Image.Resampling.LANCZOS),
                    }