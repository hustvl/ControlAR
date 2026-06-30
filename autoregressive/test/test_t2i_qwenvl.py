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
# from autoregressive.models.gpt import GPT_models
from autoregressive.models.gpt_edit import GPT_models
from autoregressive.models.generate_qwenvl import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# from dataset.t2i_control import build_t2i_control_code
from accelerate import Accelerator
# from dataset.build import build_dataset
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
# from condition.canny import CannyDetector
# from condition.hed import HEDdetector
import numpy as np
from PIL import Image
# from condition.lineart import LineArt
import cv2
# from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel
from evaluations.metrics_calculator import MetricsCalculator
from torch.utils.data import Dataset
import PIL
import torch.distributed as dist
import json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv

def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="ssim_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,tgt_mask)


class PIE_Bench_Dataset(Dataset):
    '''
    '''
    def __init__(self,
                 args,
                 dataset_path,
                 llm_tokenizer,
                 mode='train',
                 ):

        self.args = args
        self.qwenvl_image_size = args.qwenvl_image_size
        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path, 'mapping_file.json'), 'r') as file:
            self.dataset = json.load(file)

        self.mapping = {}
        for idx, key in enumerate(self.dataset.keys()):
            self.mapping[idx] = key
        
        # LLM tokenizer
        self.llm_tokenizer = llm_tokenizer


    def __len__(self,):
        return len(self.dataset)

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

    def mask_decode(self, encoded_mask, image_shape=[512,512]):
        length=image_shape[0]*image_shape[1]
        mask_array=np.zeros((length,))
        
        for i in range(0,len(encoded_mask),2):
            splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
            for j in range(splice_len):
                mask_array[encoded_mask[i]+j]=1
                
        mask_array=mask_array.reshape(image_shape[0], image_shape[1])
        # to avoid annotation errors in boundary
        mask_array[0,:]=1
        mask_array[-1,:]=1
        mask_array[:,0]=1
        mask_array[:,-1]=1
                
        return mask_array
    
    def collate_fn(self, examples):
        images = [example["image"] for example in examples]
        image_path = [example["image_path"] for example in examples]
        input_img = torch.stack([example["input_img"] for example in examples])
        edited_img = torch.stack([example["edited_img"] for example in examples])
        _input_img = [example["_input_img"] for example in examples]
        _original_prompt = [example["_original_prompt"] for example in examples]
        _editing_prompt = [example["_editing_prompt"] for example in examples]
        mask = [example["_mask"] for example in examples]
        prompts = [example["_edit_txt"] for example in examples]
    
        return {
                'image_path': image_path,
                'input_img': input_img,
                'edited_img': edited_img,
                "prompts": prompts,
                "images": images,
                "_input_img":_input_img,
                "_original_prompt": _original_prompt,
                "_editing_prompt": _editing_prompt,
                "_mask": mask,
                }

    def __getitem__(self, index):

        # ['image_path', 'original_prompt', 'editing_prompt', 'editing_instruction', 'editing_type_id', 'blended_word', 'mask']
        
        data_path = self.dataset[self.mapping[index]]['image_path']

        input_img = Image.open(os.path.join(self.dataset_path, 'annotation_images', data_path)).convert('RGB')
        _input_img = input_img
        _mask = self.mask_decode(self.dataset[self.mapping[index]]["mask"])
        _mask = _mask[:,:,np.newaxis].repeat([3],axis=2)
        _original_prompt = self.dataset[self.mapping[index]]["original_prompt"].replace("[", "").replace("]", "")
        _editing_prompt = self.dataset[self.mapping[index]]["editing_prompt"].replace("[", "").replace("]", "")

        edit_txt = self.dataset[self.mapping[index]]['editing_instruction']

        image = input_img
        input_img = self._whiten_transparency(input_img)
        input_img = self._vqgan_input_from(input_img)
        edited_img = - torch.ones(input_img.shape)

        # save_input_img = (input_img.permute(1,2,0)+1)/2 * 255.
        # save_input_img = Image.fromarray(np.array(save_input_img).astype(np.uint8))
        # save_input_img.save(f"{self.args.gpt_ckpt[:-3]}/PIE-bench/input/PIE_{index:08d}_input.png")
        # Create the content to be saved
        # content = f"Edited Text:\n{edit_txt}"
        # # Save the content to the file
        # with open(f'{self.args.gpt_ckpt[:-3]}/PIE-bench/text/PIE_{index:08d}_txt.txt', "w") as file:
        #     file.write(content)

        return {
                'image_path': data_path,
                'image': image.resize((self.qwenvl_image_size,self.qwenvl_image_size)),
                'input_img': input_img,
                'edited_img': edited_img,
                '_input_img': np.array(_input_img),
                '_original_prompt': _original_prompt,
                '_editing_prompt': _editing_prompt,
                '_mask': np.array(_mask),
                '_edit_txt': edit_txt,
                }






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
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        condition_type=args.condition_type,
        adapter_size=args.adapter_size,
        encoder_name=args.encoder_name,
        qwenvl_image_size=args.qwenvl_image_size,
        use_connector=args.use_connector,
        use_adapter=args.use_adapter,
        use_fusion=args.use_fusion,
        adapter_name=args.adapter_name,
        use_prefilling=args.use_prefilling,
    ).to(device=device, dtype=precision)

    _, file_extension = os.path.splitext(args.gpt_ckpt)
    if file_extension.lower() == '.safetensors':
        from safetensors.torch import load_file
        model_weight = load_file(args.gpt_ckpt)
        gpt_model.load_state_dict(model_weight, strict=True)
        gpt_model.eval()
    else:
        checkpoint = torch.load(args.gpt_ckpt,weights_only=False, map_location="cpu")
        if "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint: # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        
       
        
        # import pdb;pdb.set_trace()
        gpt_model.load_state_dict(model_weight, strict=True)
        # import pdb;pdb.set_trace()
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
    
    qwenvl_processor = AutoProcessor.from_pretrained(args.qwenvl_weight)
    if args.qwenvl_name == "2.5":
        qwenvl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
    else:
        qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
    qwenvl_model.to(device)

    if args.qwenvl_lora_path is not None:
        qwenvl_lora_model = PeftModel.from_pretrained(
            qwenvl_model,
            args.qwenvl_lora_path,  
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        qwenvl_lora_model = qwenvl_model
    qwenvl_lora_model.eval()
    
    dataset = PIE_Bench_Dataset(args, dataset_path='ControlAR/evaluations/PIE-Bench', llm_tokenizer=t5_model.tokenizer, mode='val')
    
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
    
    evaluation_result = []
    metrics = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
                "ssim_unedit_part", "clip_similarity_source_image",
                "clip_similarity_target_image", "clip_similarity_target_image_edit_part",
                ]
    metrics_calculator = MetricsCalculator(device)
    save_path = os.path.join('ControlAR/sample/PIE-Bench', os.path.basename(os.path.dirname(os.path.dirname(args.gpt_ckpt))), os.path.splitext(os.path.basename(args.gpt_ckpt))[0])
    if rank == 0:
        os.makedirs(save_path + '/input', exist_ok=True)
        os.makedirs(save_path + '/text', exist_ok=True)
        os.makedirs(save_path + '/edit', exist_ok=True)
        os.makedirs(save_path + f'/visualization/txt_{args.cfg_scale}/', exist_ok=True)
        os.makedirs(save_path + f'/samples/txt_{args.cfg_scale}/', exist_ok=True)
    
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total = 0
    
    for idx, batch in enumerate(loader):
        images = batch["images"]
        prompts = batch['prompts']
        source_image = batch['input_img'].to(device)
        # import pdb;pdb.set_trace()
        with torch.no_grad():
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
                # new_emb_masks = None
            c_indices = new_caption_embs * new_emb_masks[:,:, None]
            c_emb_masks = new_emb_masks
            qzshape = [len(c_indices), args.codebook_embed_dim, args.image_H//args.downsample_size, args.image_W//args.downsample_size]
            t1 = time.time()
            
            
            qwenvl_outputs_embeddings = []
            padding_masks = []
            for prompt, image in zip(prompts, images):
                messages = []
                # prompt = f"Given an image editing instruction: '{prompt},' please locate the object that needs to be edited within the image."
                # prompt = f"Please describe in great detail the details and location of the object to be edited in the image according to this editing prompt: '{prompt}."
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                        "type": "image",
                        },  
                    ],
                }]
                messages.append(qwenvl_processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                ))
                
                
                
                qwenvl_inputs = qwenvl_processor(
                    text=messages,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                )
                qwenvl_inputs = qwenvl_inputs.to(device)
                qwenvl_outputs = qwenvl_lora_model.forward(
                    **qwenvl_inputs,
                    output_hidden_states=True
                )
                qwenvl_outputs_embedding = qwenvl_outputs['hidden_states'][-1][0, -256:]#[:,-(args.qwenvl_image_size // 32)**2:]
                
                padding_mask = torch.zeros(306, dtype=torch.bool, device=qwenvl_outputs_embedding.device)
                # if qwenvl_outputs_embedding.shape[0] >= 306:
                #     qwenvl_outputs_embedding = qwenvl_outputs_embedding[-306:]
                # else:
                #     pad_length = 50 - qwenvl_outputs_embedding.shape[0] + 256
                #     qwenvl_outputs_embedding = torch.cat((torch.zeros_like(qwenvl_outputs_embedding)[-pad_length:], qwenvl_outputs_embedding), dim=0)
                #     padding_mask[:pad_length] = True
                
                qwenvl_outputs_embeddings.append(qwenvl_outputs_embedding)
                padding_masks.append(padding_mask) 
            
            qwenvl_outputs_embeddings = torch.stack(qwenvl_outputs_embeddings)
            padding_masks = torch.stack(padding_masks)
            
                 
            
            
            
            # import pdb;pdb.set_trace()
            index_sample = generate(
                gpt_model, c_indices, (args.image_H//args.downsample_size)*(args.image_W//args.downsample_size),
                c_emb_masks, 
                cfg_scale=args.cfg_scale,cfg_interval=-1,
                condition=None, qwenvl_embedding=qwenvl_outputs_embeddings, source_image=source_image.to(dtype=precision), padding_mask=None,#padding_masks,
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
            # import pdb;pdb.set_trace()
            for i in range(len(images)):
                mask = batch['_mask'][i]#.cpu().numpy() # (512, 512, 3) [0, 255]
                src_image = batch['_input_img'][i]#.cpu().numpy()
                src_image = Image.fromarray(np.uint8(src_image))
                samples = torch.clamp(samples, -1, 1)
                tgt_image = (samples[i].cpu().permute(1,2,0)+1)/2 * 255.
                tgt_image = Image.fromarray(np.uint8(tgt_image))
                src_prompt = batch['_original_prompt'][i]
                tgt_prompt = batch['_editing_prompt'][i]

                evaluation_result = [batch['image_path'][i]]
                for metric in metrics:
                    cal_metric = calculate_metric(metrics_calculator,
                                                    metric,
                                                    src_image=src_image,
                                                    tgt_image=tgt_image,
                                                    src_mask=mask,
                                                    tgt_mask=mask, src_prompt=src_prompt, tgt_prompt=tgt_prompt)
                    
                    print(metric, cal_metric)
                    evaluation_result.append(cal_metric)
                index = i * dist.get_world_size() + rank + total
                with open(save_path+f'/samples/txt_{args.cfg_scale}/PIE_Bench_{rank}.csv','a+',newline="") as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(evaluation_result)

                save_image(samples[i], f"{save_path}/samples/txt_{args.cfg_scale}/{index}_sample_txt_{args.cfg_scale}.png", nrow=4, normalize=True, value_range=(-1, 1))
                print(f"image is saved to {save_path}/samples/txt_{args.cfg_scale}/{index}_sample_txt_{args.cfg_scale}.png")

                vis_input_img = batch['input_img'][i]
                vis_input_img = (vis_input_img.permute(1,2,0) + 1) / 2.
                vis_edited_img = batch['edited_img'][i]
                vis_edited_img = (vis_edited_img.permute(1,2,0) + 1) / 2.
                vis_text = batch['prompts'][i]
                vis_sample_img = (samples[i].cpu().permute(1, 2, 0) + 1) / 2.
                vis_img = torch.cat([vis_input_img, vis_sample_img], dim=1)
                # save_image(vis_img, f"{save_path}/visualization/txt_{args.cfg_scale}/{index}_sample_txt_{args.cfg_scale}.png", nrow=4, normalize=True, value_range=(-1, 1))
                plt.imshow(vis_img)
                plt.title(vis_text, fontsize=8)
                plt.savefig(f"{save_path}/visualization/txt_{args.cfg_scale}/{index}.png", dpi=512)
                plt.clf()
            total += global_batch_size



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='model_weight/ControlAR/pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--qwenvl-weight", type=str, default='model_weight/ControlAR/pretrained_models/Qwen3-VL-2B-Instruct', help="ckpt path for vq model")
    parser.add_argument("--qwenvl-lora-path", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768], default=512)
    parser.add_argument("--image-H", type=int, default=512)
    parser.add_argument("--image-W", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--cfg-scale", type=float, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--condition-type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth', 'canny_base', 'edit'], default="edit")
    parser.add_argument("--prompt", type=str, default='a high-quality image')
    parser.add_argument("--condition-path", type=str, default='condition/example/t2i/multigen/landscape.png')
    parser.add_argument("--adapter-size", type=str, default='small')

    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--encoder-name", type=str, default='qwen', choices=["dino", "vqgan", "qwen"]) 
    parser.add_argument("--qwenvl-image-size", type=int, choices=[512, 1024], default=512)
    parser.add_argument("--use-connector", type=bool, default=False)
    parser.add_argument("--per-proc-batch-size", type=int, default=14)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--use-adapter", type=bool, default=False)
    parser.add_argument("--use-fusion", type=bool, default=False)
    parser.add_argument("--use-prefilling", type=bool, default=False)
    parser.add_argument("--adapter-name", type=str, default='dinov2', choices=["dinov2", "sam2", 'vtp']) 
    parser.add_argument("--qwenvl-name", type=str, default='3', choices=["2.5", "3"]) 
    args = parser.parse_args()
    main(args)
