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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import Accelerator
from pathlib import Path
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
import numpy as np
from PIL import Image
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
from transformers import AutoModelForCausalLM

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from autoregressive.models.Janus_t2i import Janus_models
from peft import LoraConfig, get_peft_model
from condition.hed import HEDdetector
from condition.lineart import LineArt
from condition.canny import CannyDetector
import numpy as np
from PIL import Image
from condition.lineart import LineArt
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation


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
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    
    model_path = "model_weight/ControlAR/pretrained_models/Janus-Pro-1B"
    janus_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    janus_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    janus_gpt = janus_gpt.to(device)
    
    
    
    
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
        # import pdb;pdb.set_trace()
        model.eval()
        del checkpoint
    print(f"gpt model is loaded")
    # import pdb;pdb.set_trace()
    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no need to compile model in demo") 
    
    
    

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
        get_control = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large").to(device)
    with torch.no_grad():
        
        condition_path = args.condition_path
        if args.condition_type == 'seg':
            condition_img = torch.from_numpy(np.array(Image.open(condition_path).resize((args.image_size, args.image_size))))
            condition_img = condition_img.permute(2,0,1).unsqueeze(0).repeat(8,1,1,1)
        elif args.condition_type == 'canny':
            condition_img = get_control(np.array(Image.open(condition_path).resize((args.image_size, args.image_size))))
            condition_img = torch.from_numpy(condition_img[None,None,...]).repeat(8,3,1,1)
        elif args.condition_type == 'hed':
            condition_img = get_control(torch.from_numpy(np.array(Image.open(condition_path).resize((args.image_size, args.image_size)))).permute(2,0,1).unsqueeze(0).to(device))
            condition_img = condition_img.unsqueeze(1).repeat(8,3,1,1)
        elif args.condition_type == 'lineart':
            condition_img = get_control(torch.from_numpy(np.array(Image.open(condition_path).resize((args.image_size, args.image_size)))).permute(2,0,1).unsqueeze(0).to(device).float())
            # condition_img = 1 - condition_img
            condition_img = condition_img.repeat(8,3,1,1) * 255
        elif args.condition_type == 'depth':
            images = Image.open(condition_path).resize((args.image_size, args.image_size))
            inputs = processor(images=images, return_tensors="pt", size=(args.image_size, args.image_size)).to(device)
            outputs = get_control(**inputs)
            condition_img = outputs.predicted_depth
            condition_img = condition_img.unsqueeze(0).repeat(8,3,1,1)
            condition_img = (condition_img * 255 / condition_img.max())
        condition_img = condition_img.to(device)
        condition_img = 2*(condition_img/255 - 0.5)

        prompts = [args.prompt if args.prompt is not None else "a high-quality image"]
        prompts = prompts * 8

        # condition_type_list = {'seg':'segmentation map', 'canny':'canny edge image', 'hed':'hed edge image', 'lineart':'lineart edge image', 'depth':'depth image'}
        # prompts[0] = f'Given the {condition_type_list[args.condition_type]}, generate an image following the description:' + prompts[0]
        conversation = [
            {
                "role": "User",
                "content": prompts[0],
            },
            {"role": "Assistant", "content": ""},
        ]

        sft_format = janus_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=janus_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = [sft_format + janus_chat_processor.image_start_tag] * 8
        
        output = generate(
            # janus_gpt=janus_gpt,
            mmgpt=model, 
            vl_chat_processor=janus_chat_processor,
            prompts=prompt,
            parallel_size=8,
            cfg_weight=args.cfg_scale,
            temperature=args.temperature,
            source_image=condition_img,
        )
        output = torch.clamp(output, -1, 1)

        # samples = torch.cat((condition_img[0:1].cpu(), output), dim=0)
        samples = output[0].unsqueeze(0)
        save_image(samples, f"sample/example/sample_t2i_{args.condition_type}.png", nrow=4, normalize=True, value_range=(-1, 1))
        print(f"image is saved to sample/example/sample_t2i_{args.condition_type}.png")
        print(prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    # parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--qwenvl-weight", type=str, default='model_weight/ControlAR/pretrained_models/Qwen3-VL-2B-Instruct', help="ckpt path for vq model")
    parser.add_argument("--qwenvl-lora-path", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="t2i", help="class->image or text->image")  
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)
    # parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768], default=384)
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
    parser.add_argument("--use-adapter", type=bool, default=False)
    parser.add_argument("--use-fusion", type=bool, default=False)
    parser.add_argument("--adapter-name", type=str, default='dinov2', choices=["dinov2", 'sam2']) 
    parser.add_argument("--text-length", type=int, default=50)
    args = parser.parse_args()
    main(args)
