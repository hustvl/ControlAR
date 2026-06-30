# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
import inspect
import yaml
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset, build_edit_dataset
from autoregressive.models.gpt_edit import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
import time
from language.t5 import T5Embedder
from autoregressive.models.qwenvl_gpt_edit import QwenVLGPTEdit
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM

from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from autoregressive.models.Janus_edit import JanusEdit
from peft import LoraConfig, get_peft_model

def collate_fn(examples):
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
    
def collate_fn_code(examples):
    images = [example["image"] for example in examples]
    index = [example["index"] for example in examples]
    mode = [example["mode"] for example in examples]
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    # input_ids_attn_mask = torch.stack([example["input_ids_attn_mask"] for example in examples])
    input_img = torch.stack([example["input_img"] for example in examples])
    edited_img = torch.stack([example["edited_img"] for example in examples])
    captions = [example["caption"] for example in examples]
    attn_mask = torch.stack([example["attn_mask"] for example in examples])
    qwenvl_embedding = torch.stack([example["qwenvl_embedding"] for example in examples])
    padding_mask = torch.stack([example["padding_mask"] for example in examples])
    caption_embed = torch.stack([example["caption_embed"] for example in examples])
    input_code = torch.stack([example["input_code"] for example in examples])
    edited_code = torch.stack([example["edited_code"] for example in examples])
    return {
            'index': index,
            'mode': mode,
            # 'input_ids': input_ids,
            # 'input_ids_attn_mask': input_ids_attn_mask,
            'input_img': input_img,
            'edited_img': edited_img,
            "captions": captions,
            "attn_mask": attn_mask,
            "images": images,
            "qwenvl_embedding": qwenvl_embedding,
            "padding_mask": padding_mask,
            "caption_embed": caption_embed,
            "input_code": input_code,
            "edited_code": edited_code,
            }


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


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



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)


    # Setup an experiment folder:
    time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_dir = f"{args.output_dir}/{time_record}/checkpoints"
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(f"{args.output_dir}/{time_record}")
        logger.info(f"Experiment directory created at {args.output_dir}/{time_record}")
        args_dict = vars(args)
        yaml_file = f"{args.output_dir}/{time_record}/config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(args_dict, f, indent=4, sort_keys=False)

    else:
        logger = create_logger(None)
    

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup tokenizer
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Setup model
    latent_size = args.image_size // args.downsample_size

    model_path = "model_weight/ControlAR/pretrained_models/Janus-Pro-1B"
    janus_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    janus_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    janus_gpt = janus_gpt.to(device)
    
    

    
    
    janus_gpt.vision_model.requires_grad_(False)
    janus_gpt.gen_vision_model.requires_grad_(False)
    janus_gpt.aligner.requires_grad_(False)
    janus_gpt.gen_aligner.requires_grad_(False)
    janus_gpt.gen_embed.requires_grad_(False)
    # finetune these two model
    janus_gpt.language_model.requires_grad_(True)
    janus_gpt.gen_head.requires_grad_(True)
    
    

    
    
    

    logger.info(f"GPT Parameters: {sum(p.numel() for p in janus_gpt.parameters()):,}")
    
    if args.use_code:
        processor = None
        peft_model = None
    else:
        processor = AutoProcessor.from_pretrained(args.qwenvl_weight)

        qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
        qwenvl_model.enable_input_require_grads()  # 
        qwenvl_model.to(device)
        for param in qwenvl_model.parameters():
            param.requires_grad = False

        # # 获取LoRA模型
        peft_model = qwenvl_model
    model = JanusEdit(args, model=janus_gpt)#language_model=janus_gpt.language_model, gen_head=janus_gpt.gen_head)
    
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name} (shape: {param.shape})")
            # trainable_found = True
    
    
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    dataset = build_edit_dataset(args, llm_tokenizer=t5_model.tokenizer if not args.use_code else None)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_code if args.use_code else collate_fn,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    train_steps = 0
    start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_llm_loss = 0
    running_distill_loss = 0
    start_time = time.time()

    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            if args.use_code:
                input_img = batch['input_img'].to(device, non_blocking=True)
                edited_img = batch['edited_img'].to(device, non_blocking=True)
                qwenvl_embedding = batch['qwenvl_embedding'].to(device, non_blocking=True)
                padding_mask = None#batch['padding_mask'].to(device, non_blocking=True)
                # import pdb;pdb.set_trace()
                with torch.no_grad():
                    b = edited_img.shape[0]
                    edited_img_indices = janus_gpt.gen_vision_model.encode(edited_img)[2][2].reshape(b, -1)
                    # Process text for each item in batch
                    text_tokens = []
                    random_p = torch.rand(size=(b,))
                    for drop_p, text in zip(random_p, batch['captions']):
                        # text_token = torch.ones(args.text_length) * janus_chat_processor.pad_id
                        # text_token = text_token.to(torch.long)
                        conversation = [
                            {
                                "role": "User",
                                "content": text
                            },
                            {"role": "Assistant", "content": ""},
                        ]

                        sft_format = janus_chat_processor.apply_sft_template_for_multi_turn_prompts(
                            conversations=conversation,
                            sft_format=janus_chat_processor.sft_format,
                            system_prompt="",
                        )
                        text = sft_format + janus_chat_processor.image_start_tag# + janus_chat_processor.image_end_tag
                        ids = janus_chat_processor.tokenizer.encode(text)
                        ids = torch.LongTensor(ids)
                        # import pdb;pdb.set_trace()
                        try:
                            ids = repeat_pad(ids, pad_id=janus_chat_processor.pad_id, start=3, end=len(ids)-5, text_length=args.text_length)
                        except Exception as e:
                            ids = torch.ones(args.text_length) * janus_chat_processor.pad_id
                            ids = ids.to(torch.long)
                        
                        if drop_p < 0.1:
                            # ids[1:-1] = janus_chat_processor.pad_id
                            ids[1:] = janus_chat_processor.pad_id
                        if len(ids) > args.text_length:
                            # ids = ids[:args.text_length]
                            ids = torch.cat((ids[:args.text_length-1], ids[-1:]), dim=0)
                        # text_token[-len(ids):] = ids
                        text_token = ids
                        text_tokens.append(text_token)

                    img_embeds = janus_gpt.prepare_gen_img_embeds(edited_img_indices) 
                    lengths = list([len(t) for t in text_tokens])
                    # lengths = list([args.text_length for t in text_tokens])
                    max_len = max(lengths) + 576
                    max_len = args.text_length + 576# + 1
                    embed_dim = img_embeds.shape[-1]
                    
                    inputs_embeds = torch.zeros(b, max_len, embed_dim, device = img_embeds.device, dtype=img_embeds.dtype)
                    attention_mask = torch.zeros(b, max_len, device = img_embeds.device, dtype=torch.bool)
                    for i in range(b):
                        text_embeds = janus_gpt.language_model.get_input_embeddings()(text_tokens[i].to(img_embeds.device))
                        text_seqlen = text_embeds.shape[0]
                        # inputs_embeds[i, 0:text_seqlen-1] = text_embeds[:-1]
                        # inputs_embeds[i, text_seqlen-1: text_seqlen-1 + 576] = img_embeds[i]
                        # inputs_embeds[i, text_seqlen - 1 + 576] = text_embeds[-1]
                        # attention_mask[i, :text_seqlen + 576] = True
                        
                        inputs_embeds[i, args.text_length-text_seqlen:args.text_length] = text_embeds#[:-1]
                        inputs_embeds[i, args.text_length:args.text_length + 576] = img_embeds[i]
                        # inputs_embeds[i, args.text_length + 576] = text_embeds[-1]
                        # attention_mask[i, args.text_length-text_seqlen:args.text_length + 576] = True
                        attention_mask[i, :args.text_length + 576] = True
                        
                        
                        
                    
                    position_ids = torch.arange(0, max_len).unsqueeze(0).to(device=img_embeds.device, dtype = torch.long)
            else:
                input_img = batch['input_img'].to(device, non_blocking=True)
                edited_img = batch['edited_img'].to(device, non_blocking=True)
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                input_ids_attn_mask = batch['input_ids_attn_mask'].to(device, non_blocking=True)
                feat_lens = batch['feat_len']
                # process text ids to embeddings
                with torch.no_grad():
                    input_txt_embs = t5_model.model(
                        input_ids=input_ids,
                        attention_mask=input_ids_attn_mask,
                    )['last_hidden_state'].detach()
                new_emb_masks = batch['emb_mask'].to(device, non_blocking=True)
                new_input_txt_embs = []
                for idx, (input_txt_emb, feat_len) in enumerate(zip(input_txt_embs, feat_lens)):
                    new_input_txt_emb = torch.cat([input_txt_emb[feat_len:], input_txt_emb[:feat_len]])
                    new_input_txt_embs.append(new_input_txt_emb)
                new_input_txt_embs = torch.stack(new_input_txt_embs).to(device, non_blocking=True)
                
                input_txt_embs = new_input_txt_embs * new_emb_masks[:,:, None]
                
                # process image ids to embeddings
                with torch.no_grad():
                    # if args.encoder_name == 'vqgan':
                    # _, _, [_, _, input_img_indices] = vq_model.encode(input_img)
                    # input_img_indices = input_img_indices.reshape(input_img.shape[0], -1)
                    # input_img = input_img_indices
                    _, _, [_, _, edited_img_indices] = vq_model.encode(edited_img)
                    edited_img_indices = edited_img_indices.reshape(edited_img.shape[0], -1)
                qwenvl_embedding = None
            # attn_mask = attention_mask
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                loss, _ = model(inputs_embeds=inputs_embeds, edited_img_indices=edited_img_indices, attention_mask=attention_mask, source_image=input_img, qwenvl_embedding=qwenvl_embedding, position_ids=position_ids, lengths=torch.Tensor(lengths), padding_mask=padding_mask)
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_llm_loss = 0
                running_distill_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "steps": train_steps,
                        "args": args
                    }
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        lora_path = f"{checkpoint_dir}/{train_steps:07d}_lora"
                        # language_model.save_pretrained(lora_path)
                    
                dist.barrier()


    if rank==0:
        # output COMPLETE
        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, 'COMPLETE')
        # Create the empty file
        with open(file_path, 'w') as f:
            pass  # Just create an empty file and close it

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='./pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--qwenvl-weight", type=str, default='./pretrained_models/vq_ds16_t2i.pt', help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i', 'edit'], default="edit")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--output-dir", type=str, default="checkpoints/test")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--qwenvl-image-size", type=int, choices=[512, 1024], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=24)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--dataset-list", nargs='+', default=['seedxunsplash', 'pipe', 'pico'])
    parser.add_argument("--pipe-path", type=str, default='/workspace/dataset/SmartEdit/PIPE_HF')
    parser.add_argument("--pipe-prob", type=float, default=0.7)
    parser.add_argument("--seedxunsplash-path", type=str, default='/workspace/dataset/SmartEdit/Seedx_Unsplash_HF')
    parser.add_argument("--seedxunsplash-prob", type=float, default=1.0)
    parser.add_argument("--condition-type", type=str, choices=['canny', 'hed', 'lineart', 'depth', 'edit'], default="edit")
    parser.add_argument("--encoder-name", type=str, choices=['dino', 'vqgan', 'qwen'], default="vqgan")
    parser.add_argument("--use-connector", type=bool, default=False)
    parser.add_argument("--use-adapter", type=bool, default=False)
    parser.add_argument("--adapter-size", type=str, choices=['small', 'base'], default="small")
    parser.add_argument("--use-fusion", type=bool, default=False)
    parser.add_argument("--use-code", type=bool, default=False)
    parser.add_argument("--adapter-name", type=str, choices=['dinov2', 'sam2'], default="dinov2")
    parser.add_argument("--text-length", type=int, default=50, help="max token number of condition input")
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    args = parser.parse_args()
    main(args)