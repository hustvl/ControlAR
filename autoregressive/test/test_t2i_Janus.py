import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
# from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")
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
from dataset.t2i_control import build_t2i_control
from accelerate import Accelerator
# from dataset.build import build_dataset
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
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
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
from transformers import AutoModelForCausalLM

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
# from autoregressive.models.Janus_edit import JanusEdit
from autoregressive.models.Janus_t2i import Janus_models
from tqdm import tqdm

from typing import Any, BinaryIO, Optional, Union
import pathlib
@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, list[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid



@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, list[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    size: Optional[int] = 512,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr).resize((size,size), Image.BICUBIC)
    im.save(fp, format=format)


def repeat_pad(ids, pad_id, start, end, text_length=50):
    l = len(ids)
    if l >= text_length:
        return ids
    pad_ids = torch.Tensor([pad_id])
    pad_ids = pad_ids.long()
    text_ids = ids[start:end]
    start_ids = ids[:start]
    end_ids = ids[end:]
    new_ids = ids[start:]
    while len(new_ids) < text_length-len(start_ids)-len(text_ids)-1:
        new_ids = torch.cat((text_ids, pad_ids, new_ids), dim=0)
    if len(new_ids) == text_length-len(start_ids):
        new_ids = torch.cat((start_ids, new_ids), dim=0)
    else:
        new_ids = torch.cat((start_ids, text_ids[:text_length-len(new_ids)-len(start_ids)-1], pad_ids, new_ids), dim=0)
    return new_ids.long()


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    source_image: torch.Tensor,
    prompts: str,
    temperature: float = 0.7,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    text_length: int = 50,
    
):
    # import pdb;pdb.set_trace()
    input_idss = []
    for prompt in prompts:
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        # print(len(input_ids))
        input_ids = repeat_pad(input_ids, pad_id=vl_chat_processor.pad_id, start=3, end=len(input_ids)-5, text_length=text_length)
        
        if len(input_ids) > text_length:
            input_ids = torch.cat((input_ids[:text_length-5], input_ids[-5:]), dim=0)
        input_idss.append(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_idss[i//2]
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    # import pdb;pdb.set_trace()
    inputs_embeds = mmgpt.model.language_model.get_input_embeddings()(tokens)
    # import pdb;pdb.set_trace()

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    source_image_embedding = mmgpt.adapter(source_image)
    source_image_embedding = mmgpt.condition_mlp(source_image_embedding)
    
    condition_tokens = torch.zeros_like(source_image_embedding)
    condition_tokens = torch.cat((condition_tokens,condition_tokens),dim=0)
    
    for i in range(parallel_size*2):
        if i % 2 == 0:
            condition_tokens[i] = source_image_embedding[i//2]

    
    # import pdb;pdb.set_trace()
    for i in range(image_token_num_per_image):
        inputs_embeds[:,-1] = inputs_embeds[:,-1] + condition_tokens[:, i]# * mmgpt.control_strength
        outputs = mmgpt.model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        # hidden_states[:,-1] = hidden_states[:,-1] + condition_tokens[:, i]
        
        logits = mmgpt.model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    
    dec = mmgpt.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu()#.numpy()#.transpose(0, 2, 3, 1)
    return dec




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
    
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]

    model_path = "model_weight/ControlAR/pretrained_models/Janus-Pro-1B"
    janus_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    janus_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    janus_gpt = janus_gpt.to(device)
    
    
    if args.condition_type == 'hed':
        get_condition = HEDdetector().to(device).eval()
    elif args.condition_type == 'lineart':
        get_condition = LineArt()
        get_condition.load_state_dict(torch.load('ControlAR/condition/ckpts/model.pth', map_location=torch.device('cpu')))
        get_condition.to(device)
    
    model = Janus_models(args, model=janus_gpt).to(device)

    
    _, file_extension = os.path.splitext(args.gpt_ckpt)
    if file_extension.lower() == '.safetensors':
        from safetensors.torch import load_file
        model_weight = load_file(args.gpt_ckpt)
        model.load_state_dict(model_weight, strict=False)
        model.eval()
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
        model.load_state_dict(model_weight, strict=True)
        model.eval()
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
    
    
    dataset = build_t2i_control(args, val=True)
    
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
    
    os.makedirs(f"{args.sample_dir}/visualization", exist_ok=True)
    os.makedirs(f"{args.sample_dir}/annotations", exist_ok=True)
    
    total_len = len(loader)
    group_len = total_len // 4
    total = 0
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    condition_type_list = {'seg':'segmentation map', 'canny':'canny edge image', 'hed':'hed edge image', 'lineart':'lineart edge image', 'depth':'depth image'}
    for idx, batch in enumerate(tqdm(loader)):
        # if idx > 100:
        #     break
        # if idx < args.group_id * group_len or idx >= (args.group_id + 1) * group_len:
        #     continue
        image = batch['image'].to(device, non_blocking=True)
        bs = image.shape[0]
        condition_img = batch['control'].to(device, non_blocking=True)
        condition_img_origin = batch['control_origin'].to(device, non_blocking=True)
        if args.condition_type == 'seg':
            label = batch['label']
        with torch.no_grad():
            prompts = []
            for prompt in batch['caption']:
                
                prompt = f'Given the {condition_type_list[args.condition_type]}, generate an image following the description:' + prompt
                conversation = [
                    {
                        "role": "User",
                        "content": prompt,
                    },
                    {"role": "Assistant", "content": ""},
                ]

                sft_format = janus_chat_processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversation,
                    sft_format=janus_chat_processor.sft_format,
                    system_prompt="",
                )
                prompts.append(sft_format + janus_chat_processor.image_start_tag)
            
            if args.condition_type in ['seg', 'depth']:
                condition_img = condition_img
                
                condition_img_origin = condition_img_origin
            elif args.condition_type == 'hed':
                condition_img = get_condition(condition_img).unsqueeze(1).repeat(1,3,1,1)
                condition_img = 2*(condition_img/255 - 0.5)
                
                condition_img_origin = get_condition(condition_img_origin).unsqueeze(1).repeat(1,3,1,1)
                condition_img_origin = 2*(condition_img_origin/255 - 0.5)
            elif args.condition_type == 'lineart':
                condition_img = get_condition(condition_img.float()).repeat(1,3,1,1)
                condition_img = 2*((1-condition_img) - 0.5)
                
                condition_img_origin = get_condition(condition_img_origin.float()).repeat(1,3,1,1)
                condition_img_origin = 2*(condition_img_origin - 0.5)
        
            output = generate(
                mmgpt=model, 
                vl_chat_processor=janus_chat_processor,
                prompts=prompts,
                parallel_size=bs,
                temperature=args.temperature,
                cfg_weight=args.cfg_scale,
                source_image=condition_img.to(dtype=precision, device=device),
                )
            samples = torch.clamp(output, -1, 1)
        # import pdb;pdb.set_trace()
        for i in range(len(samples)):
            image_name = batch['image_name'][i]
            # image_name = i * dist.get_world_size() + rank + total
            if args.condition_type == 'seg':
                save_image(samples[i].unsqueeze(0), f"{args.sample_dir}/visualization/{image_name}", nrow=1, normalize=True, value_range=(-1, 1), size=512)
                save_image(condition_img[i].unsqueeze(0), f"{args.sample_dir}/annotations/{image_name}", nrow=1, normalize=True, value_range=(-1, 1), size=512)
            else:
                save_image(samples[i].unsqueeze(0), f"{args.sample_dir}/visualization/{image_name}", nrow=1, normalize=True, value_range=(-1, 1), size=384)
                save_image(condition_img[i].unsqueeze(0), f"{args.sample_dir}/annotations/{image_name}", nrow=1, normalize=True, value_range=(-1, 1), size=384)
        break
        # if args.condition_type == 'seg':
        #     Image.fromarray(label[0].numpy().astype('uint8'), mode='L').save(f"{args.sample_dir}/annotations/{image_name}")
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
    parser.add_argument("--cfg-scale", type=float, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--condition-type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth', 'canny_base', 'edit'], default="edit")
    parser.add_argument("--prompt", type=str, default='a high-quality image')
    parser.add_argument("--condition-path", type=str, default='condition/example/t2i/multigen/landscape.png')
    parser.add_argument("--adapter-size", type=str, default='small')

    parser.add_argument("--get-image", type=bool, default=True)
    parser.add_argument("--get-prompt", type=bool, default=False)
    parser.add_argument("--get-label", type=bool, default=False)
    parser.add_argument("--data-folder", type=str, default="data/MultiGen20M/val")
    parser.add_argument("--data-folder2", type=str, default=None)
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--encoder-name", type=str, default='qwen', choices=["dino", "vqgan", "qwen"]) 
    parser.add_argument("--qwenvl-image-size", type=int, choices=[512, 1024], default=512)
    parser.add_argument("--use-connector", type=bool, default=False)
    parser.add_argument("--per-proc-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--use-adapter", type=bool, default=False)
    parser.add_argument("--use-fusion", type=bool, default=False)
    parser.add_argument("--adapter-name", type=str, default='dinov2', choices=["dinov2", "sam2"]) 
    parser.add_argument("--sample-dir", type=str, default='sample/canny') 
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--text-length", type=int, default=50)
    args = parser.parse_args()
    main(args)
