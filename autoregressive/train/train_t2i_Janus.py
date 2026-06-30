# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
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
import sys
import yaml
current_directory = os.getcwd()
sys.path.append(current_directory)
from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.models.Janus_t2i import Janus_models
# from tokenizer.tokenizer_image.vq_model import VQ_models
from accelerate.utils import ProjectConfiguration, set_seed
from pathlib import Path
from accelerate import Accelerator
from language.t5 import T5Embedder
from dataset.t2i_control import build_t2i_control_code, build_t2i_control, build_t2i_control_all
import torch._dynamo
from condition.hed import HEDdetector
from condition.lineart import LineArt
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM
torch._dynamo.config.suppress_errors = True


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

    # # Setup an experiment folder:
    # if rank == 0:
    #     os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    #     experiment_index = len(glob(f"{args.results_dir}/*"))
    #     model_string_name = args.gpt_model.replace("/", "-") 
    #     experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    #     checkpoint_dir = f"{experiment_dir}/checkpoints"
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     logger = create_logger(experiment_dir)
    #     logger.info(f"Experiment directory created at {experiment_dir}")

    #     time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #     cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
    #     cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
    #     os.makedirs(cloud_checkpoint_dir, exist_ok=True)
    #     logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    # else:
    #     logger = create_logger(None)
        
    # Setup an experiment folder:
    time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    checkpoint_dir = f"{args.output_dir}/ControlAR-{time_record}/checkpoints"
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(f"{args.output_dir}/ControlAR-{time_record}")
        logger.info(f"Experiment directory created at {args.output_dir}/ControlAR-{time_record}")
        args_dict = vars(args)
        yaml_file = f"{args.output_dir}/ControlAR-{time_record}/config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(args_dict, f, indent=4, sort_keys=False)

    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    
    model_path = "/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/Janus-Pro-1B"
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
    
    # for name, param in janus_gpt.language_model.named_parameters():
    #     if 'layers.0' in name:
    #         param.requires_grad = True
    #         print('可训练：')
    #         print(name)
    
    # logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.condition_type == 'hed':
        get_condition = HEDdetector().to(device).eval()
    elif args.condition_type == 'lineart':
        get_condition = LineArt()
        get_condition.load_state_dict(torch.load('/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/condition/ckpts/model.pth', map_location=torch.device('cpu')))
        get_condition.to(device)
    elif args.condition_type == 'all':
        condition_type_list = ['segmentation map', 'canny edge image', 'hed edge image', 'lineart edge image', 'depth image']
        get_hed = HEDdetector().to(device).eval()
        get_lineart = LineArt()
        get_lineart.load_state_dict(torch.load('/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/condition/ckpts/model.pth', map_location=torch.device('cpu')))
        get_lineart.to(device)
    
    

    model = Janus_models(args, model=janus_gpt)
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    if args.condition_type == 'all':
        train_dataset = build_t2i_control_all(args)
    else:
        train_dataset = build_t2i_control(args)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )

    loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    #     # optimizer.load_state_dict(checkpoint["optimizer"])
    #     train_steps = 0#checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
    #     start_epoch = 0#int(train_steps / int(len(dataset) / args.global_batch_size))
    #     train_steps = 0#int(start_epoch * int(len(dataset) / args.global_batch_size))
    #     del checkpoint
    #     logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    #     logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    # else:
    train_steps = 0
    start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    # model.zero_init_mlp()
    model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y, attn_mask, valid in loader:
        for batch in loader:
            
            image = batch['image'].to(device, non_blocking=True)
            condition_img = batch['control'].to(device, non_blocking=True).float()
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                
                if args.condition_type in ['seg', 'depth']:
                    condition_img = condition_img
                elif args.condition_type == 'hed':
                    condition_img = get_condition(condition_img).unsqueeze(1).repeat(1,3,1,1)
                    condition_img = 2*(condition_img/255 - 0.5)
                elif args.condition_type == 'lineart':
                    condition_img = get_condition(condition_img.float()).repeat(1,3,1,1)
                    condition_img = 2*(condition_img - 0.5)
                elif args.condition_type == 'all':
                    condition_type = batch['condition_type']
                    hed_mask = (condition_type == 2)  # shape: (128,), dtype: bool）
                    selected_hed_data = condition_img[hed_mask.squeeze(1)]  # shape: (n0, C, H, W), n0 = mask.sum()
                    processed_hed_data = get_hed(selected_hed_data.float()).unsqueeze(1).repeat(1,3,1,1)
                    processed_hed_data = 2*(processed_hed_data/255 - 0.5)
                    condition_img[hed_mask.squeeze(1)] = processed_hed_data
                    
                    lineart_mask = (condition_type == 3)  # shape: (128,), dtype: bool）
                    selected_lineart_data = condition_img[lineart_mask.squeeze(1)]  # shape: (n0, C, H, W), n0 = mask.sum()
                    processed_lineart_data = 1 - get_lineart(selected_lineart_data.float()).repeat(1,3,1,1)
                    processed_lineart_data = 2*(processed_lineart_data - 0.5)
                    condition_img[lineart_mask.squeeze(1)] = processed_lineart_data
                    
                # import pdb;pdb.set_trace()
                b = image.shape[0]
                img_indices = janus_gpt.gen_vision_model.encode(image)[2][2].reshape(b, -1)
                # Process text for each item in batch
                text_tokens = []
                random_p = torch.rand(size=(b,))
                drop_ids = torch.zeros(b)
                index = 0
                for drop_p, text in zip(random_p, batch['caption']):
                    if args.condition_type == 'all':
                        text = f'Given the {condition_type_list[condition_type[index].item()]}, generate an image following the description:' + text
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
                    text = sft_format + janus_chat_processor.image_start_tag
                    ids = janus_chat_processor.tokenizer.encode(text)
                    ids = torch.LongTensor(ids)
                    try:
                        ids = repeat_pad(ids, pad_id=janus_chat_processor.pad_id, start=3, end=len(ids)-5, text_length=args.text_length)
                    except Exception as e:
                        ids = torch.ones(args.text_length) * janus_chat_processor.pad_id
                        ids = ids.to(torch.long)
                        print("ids 错误")
                    
                    if drop_p < 0.1:
                        ids[1:] = janus_chat_processor.pad_id
                        drop_ids[index] = 1
                    if len(ids) > args.text_length:
                        # ids = torch.cat((ids[:args.text_length-1], ids[-1:]), dim=0)
                        ids = torch.cat((ids[:args.text_length-5], ids[-5:]), dim=0)
                    text_token = ids
                    text_tokens.append(text_token)
                    index += 1
                
                

                img_embeds = janus_gpt.prepare_gen_img_embeds(img_indices) 
                lengths = list([len(t) for t in text_tokens])
                max_len = max(lengths) + 576
                max_len = args.text_length + 576# + 1
                embed_dim = img_embeds.shape[-1]
                
                
                drop_ids = drop_ids.bool().to(device = img_embeds.device)
                inputs_embeds = torch.zeros(b, max_len, embed_dim, device = img_embeds.device, dtype=img_embeds.dtype)
                attention_mask = torch.zeros(b, max_len, device = img_embeds.device, dtype=torch.bool)
                for i in range(b):
                    text_embeds = janus_gpt.language_model.get_input_embeddings()(text_tokens[i].to(img_embeds.device))
                    text_seqlen = text_embeds.shape[0]
                    
                    # inputs_embeds[i, :text_seqlen] = text_embeds#[:-1]
                    # inputs_embeds[i, text_seqlen: text_seqlen + 576] = img_embeds[i]
                    # # inputs_embeds[i, text_seqlen - 1 + 576] = text_embeds[-1]
                    # attention_mask[i, :text_seqlen + 576] = True
                
                    inputs_embeds[i, args.text_length-text_seqlen:args.text_length] = text_embeds
                    inputs_embeds[i, args.text_length:args.text_length + 576] = img_embeds[i]
                    attention_mask[i, :args.text_length + 576] = True
                
                position_ids = torch.arange(0, max_len).unsqueeze(0).to(device=img_embeds.device, dtype = torch.long)
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                loss, _ = model(inputs_embeds=inputs_embeds, target_indices=img_indices, attention_mask=attention_mask, source_image=condition_img, position_ids=position_ids, lengths=torch.Tensor(lengths), drop_ids=drop_ids)
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
                    
                    # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    # torch.save(checkpoint, cloud_checkpoint_path)
                    # logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=False)
    parser.add_argument("--t5-feat-path", type=str, required=False)
    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=False, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')

    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='t2i_control')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=28)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--gpt-ckpt", type=str, required=False)
    parser.add_argument("--data-folder", type=str, default=None)
    parser.add_argument("--data-folder2", type=str, default=None)
    parser.add_argument("--data-folder3", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/outputs")
    parser.add_argument("--condition-type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth', 'all'], default="hed")

    parser.add_argument("--get-image", type=bool, default=True)
    parser.add_argument("--get-prompt", type=bool, default=False)
    parser.add_argument("--get-label", type=bool, default=False)

    parser.add_argument("--keep_in_memory",type=bool,default=False)
    parser.add_argument("--wrong_ids_file",type=str,default=None)
    parser.add_argument("--logging_dir",type=str,default="logs")
    parser.add_argument("--adapter-size", type=str, default='small')
    parser.add_argument("--text-length", type=int, default=50, help="max token number of condition input")
    args = parser.parse_args()
    main(args)
