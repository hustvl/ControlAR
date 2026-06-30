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
import torch.nn as nn
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import math
# from natsort import natsorted
# import wandb
# os.environ["WANDB_API_KEY"] = 'WANDB_CREDENTIAL'

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
from autoregressive.models.distill_loss import Distill_Loss
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
# from feature_encoders.build import Semantic_Encoder


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


def add_lora_to_layers(model, target_module_names, r=8, lora_alpha=16, lora_dropout=0.05):
    """
    递归遍历 model，将 layers 下的指定线性层替换为 LoRALinear
    """
    for name, child in model.named_children():
        # 如果当前模块是 layers（ModuleList of TransformerBlock）
        if name == "layers":
            for layer in child:
                _replace_modules_with_lora(layer, target_module_names, r, lora_alpha, lora_dropout)
        else:
            # 递归处理子模块（但跳过非-layers 部分）
            add_lora_to_layers(child, target_module_names, r, lora_alpha, lora_dropout)

def _replace_modules_with_lora(module, target_names, r, lora_alpha, lora_dropout):
    """替换 module 中名为 target_names 的 nn.Linear 为 LoRALinear"""
    for attr_name in dir(module):
        if attr_name in target_names:
            sub_module = getattr(module, attr_name)
            if isinstance(sub_module, nn.Linear):
                print(f"Replacing {module.__class__.__name__}.{attr_name} with LoRALinear")
                lora_layer = LoRALinear(
                    base_layer=sub_module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(module, attr_name, lora_layer)
    
    # 递归子模块（比如 attention 或 mlp 是独立 nn.Module）
    for child_name, child in module.named_children():
        _replace_modules_with_lora(child, target_names, r, lora_alpha, lora_dropout)
        
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer  # 原始线性层（将被冻结）
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Dropout（可选）
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x

        # LoRA 可训练参数
        self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # 类似 Linear 初始化
        nn.init.zeros_(self.lora_B)  # B 初始为 0，保证初始输出不变

        # 冻结原始权重
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_out = self.base_layer(x)
        # LoRA 分支：x -> A -> B -> scaling
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return original_out + lora_out * self.scaling



def main(args):
    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # # Setup DDP:
    # init_distributed_mode(args)
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # seed = args.global_seed * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    # Setup an experiment folder:
    time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_dir = f"{args.output_dir}/{time_record}/checkpoints"
    
    logging_dir = checkpoint_dir# Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )


    
    
    if accelerator.is_main_process:
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
    # logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # setup tokenizer
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    if not args.use_code:
        assert os.path.exists(args.t5_path)
        t5_model = T5Embedder(
            device=accelerator.device, 
            local_cache=True, 
            cache_dir=args.t5_path, 
            dir_or_name=args.t5_model_type,
            torch_dtype=precision,
            model_max_length=args.t5_feature_max_len,
        )
    
        assert os.path.exists(args.vq_ckpt)
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(accelerator.device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint

    # Setup model
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        condition_type=args.condition_type,
        encoder_name=args.encoder_name,
        qwenvl_image_size=args.qwenvl_image_size,
        use_connector=args.use_connector,
        use_adapter=args.use_adapter,
        adapter_size=args.adapter_size,
        use_fusion=args.use_fusion,
        adapter_name=args.adapter_name,
        use_prefilling=args.use_prefilling,
    ).to(accelerator.device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")
    
    
    
    # 使用Transformers加载模型权重
    if args.use_code:
        processor = None
        peft_model = None
    else:
        processor = AutoProcessor.from_pretrained(args.qwenvl_weight)
        if args.qwen_name == "2.5":
            qwenvl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
        else:
            qwenvl_model = Qwen3VLForConditionalGeneration.from_pretrained(args.qwenvl_weight, device_map=None, torch_dtype=torch.bfloat16, trust_remote_code=True,)
        qwenvl_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
        qwenvl_model.to(accelerator.device)
        for param in qwenvl_model.parameters():
            param.requires_grad = False

        # # 获取LoRA模型
        peft_model = qwenvl_model#get_peft_model(qwenvl_model, config)
        # import pdb;pdb.set_trace()
    model = QwenVLGPTEdit(args, qwenvl_model=peft_model, processor=processor, gpt_model=gpt_model)
    
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    dataset = build_edit_dataset(args, llm_tokenizer=t5_model.tokenizer if not args.use_code else None)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        # sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn_code if args.use_code else collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    # train_dataset = LightTransferTripleDataset(args, tokenizer=tokenizer)  # light transfer

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=train_dataset.collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )
    
    logger.info(f"Dataset contains {len(dataset):,} images")

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, weights_only=False, map_location="cpu")
        model.gpt_model.load_state_dict(checkpoint["model"], strict=False)
        if "steps" in checkpoint:
            train_steps = checkpoint["steps"]
        else:
            train_steps = 0
        del checkpoint
        start_epoch = 0#int(train_steps / int(len(dataset) / args.global_batch_size))
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    
    # model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    # scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_llm_loss = 0
    running_distill_loss = 0
    start_time = time.time()

    # distill_loss_func = Distill_Loss(precision=ptdtype, device=accelerator.device)
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            # print('ok1')
            with accelerator.accumulate(model):
                if args.use_code:
                    input_img = batch['input_img'].to(accelerator.device, non_blocking=True)
                    edited_img = batch['edited_img'].to(accelerator.device, non_blocking=True)
                    qwenvl_embedding = batch['qwenvl_embedding'].to(accelerator.device, non_blocking=True)
                    input_txt_embs = batch['caption_embed'].to(accelerator.device, non_blocking=True)
                    edited_img_indices = batch['edited_code'].to(accelerator.device, non_blocking=True)
                    padding_mask = None#batch['padding_mask'].to(accelerator.device, non_blocking=True)
                else:
                    # import pdb;pdb.set_trace()
                    input_img = batch['input_img'].to(accelerator.device, non_blocking=True)
                    edited_img = batch['edited_img'].to(accelerator.device, non_blocking=True)
                    input_ids = batch['input_ids'].to(accelerator.device, non_blocking=True)
                    input_ids_attn_mask = batch['input_ids_attn_mask'].to(accelerator.device, non_blocking=True)
                    feat_lens = batch['feat_len']
                    # process text ids to embeddings
                    with torch.no_grad():
                        input_txt_embs = t5_model.model(
                            input_ids=input_ids,
                            attention_mask=input_ids_attn_mask,
                        )['last_hidden_state'].detach()
                    new_emb_masks = batch['emb_mask'].to(accelerator.device, non_blocking=True)
                    new_input_txt_embs = []
                    for idx, (input_txt_emb, feat_len) in enumerate(zip(input_txt_embs, feat_lens)):
                        new_input_txt_emb = torch.cat([input_txt_emb[feat_len:], input_txt_emb[:feat_len]])
                        new_input_txt_embs.append(new_input_txt_emb)
                    new_input_txt_embs = torch.stack(new_input_txt_embs).to(accelerator.device, non_blocking=True)
                    
                    input_txt_embs = new_input_txt_embs * new_emb_masks[:,:, None]
                    # import pdb;pdb.set_trace()
                    # process image ids to embeddings
                    with torch.no_grad():
                        _, _, [_, _, edited_img_indices] = vq_model.encode(edited_img)
                        edited_img_indices = edited_img_indices.reshape(edited_img.shape[0], -1)
                    qwenvl_embedding = None
                    padding_mask = None
                # print('ok2')
                attn_mask = batch['attn_mask']
                attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]) # (bs, n_head, seq_len, seq_len)
                # attn_mask = None
                # with torch.cuda.amp.autocast(dtype=ptdtype):  
                loss, condition_feature = model(batch['captions'], batch['images'], input_txt_embs, edited_img_indices, attn_mask, source_image=input_img, target_image=edited_img, qwenvl_embedding=qwenvl_embedding, padding_mask=padding_mask)
                # print('ok3')
                # backward pass, with gradient scaling if training in fp16         
                # scaler.scale(loss).backward()
                accelerator.backward(loss)
                # if args.max_grad_norm != 0.0:
                #     scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # # step the optimizer and scaler if training in fp16
                # scaler.step(optimizer)
                # scaler.update()
                # # flush the gradients as soon as we can, no need for this memory anymore
                # optimizer.zero_grad(set_to_none=True)
                optimizer.step()
                optimizer.zero_grad()
                # print('ok4')
                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                if accelerator.is_main_process:
                    if train_steps % args.log_every == 0:
                        # Measure training speed:
                        # torch.cuda.synchronize()
                        # print('ok5')
                        end_time = time.time()
                        steps_per_sec = log_steps / (end_time - start_time)
                        # Reduce loss history over all processes:
                        avg_loss = torch.tensor(running_loss / log_steps, device=accelerator.device)
                        # print('ok6')
                        # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item()# / dist.get_world_size()
                        logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                        # print('ok7')
                        # Reset monitoring variables:
                        running_loss = 0
                        running_llm_loss = 0
                        running_distill_loss = 0
                        log_steps = 0
                        start_time = time.time()

                # Save checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if accelerator.is_main_process:
                        if not args.no_compile:
                            model_weight = model.module._orig_mod.gpt_model.state_dict()
                        else:
                            model_weight = model.module.gpt_model.state_dict()  
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
                            # gpt_model.save_pretrained(lora_path)
                        
                    dist.barrier()


    if accelerator.is_main_process:
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
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--qwenvl-image-size", type=int, choices=[512, 1024], default=1024)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--t5-path", type=str, default='/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
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
    parser.add_argument("--adapter-name", type=str, choices=['dinov2', 'sam2', 'vtp'], default="dinov2")
    parser.add_argument("--qwen-name", type=str, choices=['2.5', '3'], default="3")
    parser.add_argument("--use-prefilling", type=bool, default=False)
    args = parser.parse_args()
    main(args)