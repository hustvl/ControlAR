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
# from tokenizer.tokenizer_image.vq_model import VQ_models
# from language.t5 import T5Embedder
# from autoregressive.models.gpt import GPT_models
# from autoregressive.models.gpt_edit import GPT_models
# from autoregressive.models.generate_qwenvl import generate
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
from autoregressive.models.Janus_edit import JanusEdit
from peft import LoraConfig, get_peft_model



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
    janus_gpt, 
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    qwenvl_embedding: torch.Tensor,
    source_image: torch.Tensor,
    prompt: str,
    temperature: float = 0.9,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    
):
    # import pdb;pdb.set_trace()
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    
    
    input_ids = repeat_pad(input_ids, pad_id=vl_chat_processor.pad_id, start=3, end=len(input_ids)-5, text_length=50)
    
    # import pdb;pdb.set_trace()
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()

    print(input_ids)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        # tokens[i,-len(input_ids):] = input_ids
        # tokens[i,:-len(input_ids)] = vl_chat_processor.pad_id
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    # import pdb;pdb.set_trace()
    inputs_embeds = janus_gpt.language_model.get_input_embeddings()(tokens)

    

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()
    
    if mmgpt.use_adapter:
        # import pdb;pdb.set_trace()
        source_image_embedding = mmgpt.adapter(source_image).to(dtype=source_image.dtype)
        if mmgpt.use_fusion:
            qwenvl_embedding = mmgpt.fusionblock(source_image_embedding, qwenvl_embedding)
        else:
            if mmgpt.qwenvl_image_size == 512:
                B, L, D = qwenvl_embedding.shape
                qwenvl_embedding = qwenvl_embedding.view(B, 16, 16, D).permute(0,3,1,2)
                qwenvl_embedding = F.interpolate(qwenvl_embedding,size=(24,24),mode='bilinear',align_corners=False).permute(0,2,3,1).flatten(1,2)

    condition_tokens = torch.cat((mmgpt.qwenvl_mlp(qwenvl_embedding.to(dtype=source_image.dtype)), source_image_embedding), dim=2)
    condition_tokens = torch.cat((condition_tokens,condition_tokens), dim=0) * mmgpt.control_strength
    
    
    # import pdb;pdb.set_trace()
    for i in range(image_token_num_per_image):
        # if i > 0:
        inputs_embeds[:,-1] = inputs_embeds[:,-1] + condition_tokens[:, i]
        outputs = janus_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = janus_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = janus_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = janus_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu()#.numpy()#.transpose(0, 2, 3, 1)

    # dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    # visual_img[:, :, :] = dec

    # os.makedirs('generated_samples', exist_ok=True)
    # for i in range(parallel_size):
    #     save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
    #     PIL.Image.fromarray(visual_img[i]).save(save_path)
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
    

    
    
    
    model = JanusEdit(args, model=janus_gpt).to(device)#language_model=janus_gpt.language_model, gen_head=janus_gpt.gen_head).to(device)
    # import pdb;pdb.set_trace()
    
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
    
    
    qwenvl_processor = AutoProcessor.from_pretrained(args.qwenvl_weight)
    qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
    # qwenvl_model.enable_input_require_grads()  #
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
        model = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large").to(device)
    with torch.no_grad():
        
        condition_path = args.condition_path
        if args.condition_type == 'seg':
            condition_img = torch.from_numpy(np.array(Image.open(condition_path)))
            condition_img = condition_img.permute(2,0,1).unsqueeze(0).repeat(2,1,1,1)
        elif args.condition_type == 'canny':
            condition_img = get_control(np.array(Image.open(condition_path)))
            condition_img = torch.from_numpy(condition_img[None,None,...]).repeat(2,3,1,1)
        elif args.condition_type == 'hed':
            condition_img = get_control(torch.from_numpy(np.array(Image.open(condition_path))).permute(2,0,1).unsqueeze(0).to(device))
            condition_img = condition_img.unsqueeze(1).repeat(2,3,1,1)
        elif args.condition_type == 'lineart':
            condition_img = get_control(torch.from_numpy(np.array(Image.open(condition_path))).permute(2,0,1).unsqueeze(0).to(device).float())
            condition_img = 1 - condition_img
            condition_img = condition_img.repeat(2,3,1,1) * 255
        elif args.condition_type == 'depth':
            images = Image.open(condition_path)
            inputs = processor(images=images, return_tensors="pt", size=(512,512)).to(device)
            outputs = model(**inputs)
            condition_img = outputs.predicted_depth
            condition_img = condition_img.unsqueeze(0).repeat(2,3,1,1)
            condition_img = (condition_img * 255 / condition_img.max())
        elif args.condition_type == 'edit':
            images = [Image.open(condition_path).convert('RGB').resize((args.qwenvl_image_size,args.qwenvl_image_size))] * 2
            condition_img = torch.from_numpy(np.array(Image.open(condition_path).resize((384, 384))))
            condition_img = condition_img.permute(2,0,1).unsqueeze(0).repeat(2,1,1,1)
        condition_img = condition_img.to(device)
        condition_img = 2*(condition_img/255 - 0.5)
        # if args.encoder_name == 'vqgan' or args.encoder_name == 'qwen':
        #     _, _, [_, _, input_img_indices] = vq_model.encode(condition_img)
        #     input_img_indices = input_img_indices.reshape(condition_img.shape[0], -1)
        #     condition_input = input_img_indices
        prompts = [args.prompt if args.prompt is not None else "a high-quality image"]
        prompts = prompts * 2

        
        messages = []
        for prompt, image in zip(prompts, images):
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
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        qwenvl_inputs = qwenvl_inputs.to(device)
        qwenvl_outputs = qwenvl_lora_model.forward(
            **qwenvl_inputs,
            output_hidden_states=True
        )
        qwenvl_outputs_embedding = qwenvl_outputs['hidden_states'][-1][:,-(args.qwenvl_image_size // 32)**2:]
        
        
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
        prompt = sft_format + janus_chat_processor.image_start_tag# + janus_chat_processor.image_end_tag
        
        output = generate(
            janus_gpt=janus_gpt,
            mmgpt=model, 
            vl_chat_processor=janus_chat_processor,
            prompt=prompt,
            parallel_size=2,
            cfg_weight=args.cfg_scale,
            qwenvl_embedding=qwenvl_outputs_embedding,
            source_image=condition_img,
            )
        # sampling_time = time.time() - t1
        # print(f"Full sampling takes about {sampling_time:.2f} seconds.")    
        # # import pdb;pdb.set_trace()
        # # loss = F.cross_entropy(index_sample.to(torch.float32), condition_input.to(torch.float32))
        # # print(f"loss: {loss}")
        # t2 = time.time()
        # print(index_sample.shape)
        # samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        # decoder_time = time.time() - t2
        # print(f"decoder takes about {decoder_time:.2f} seconds.")

        samples = torch.cat((condition_img[0:1].cpu(), output), dim=0)
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
    parser.add_argument("--qwenvl-image-size", type=int, choices=[512, 1024], default=1024)
    parser.add_argument("--use-connector", type=bool, default=False)
    parser.add_argument("--use-adapter", type=bool, default=False)
    parser.add_argument("--use-fusion", type=bool, default=False)
    parser.add_argument("--adapter-name", type=str, default='dinov2', choices=["dinov2", 'sam2']) 
    args = parser.parse_args()
    main(args)
