#!/bin/bash

torchrun --nnodes=1 --nproc-per-node=4 --master-port=12345 \
        autoregressive/train/train_qwenvl_Janus_edit.py \
        --output-dir ControlAR/outputs \
        --vq-ckpt model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt \
        --qwenvl-weight model_weight/ControlAR/pretrained_models/Qwen3-VL-2B-Instruct \
        --image-size 384 \
        --gpt-model GPT-XL \
        --gpt-ckpt model_weight/ControlAR/pretrained_models/t2i_XL_stage2_512.pt \
        --num-workers 16 \
        --epochs 5 \
        --dataset-list pipe seedxunsplash\
        --pipe-path data/PIPE \
        --pipe-prob 0.5 \
        --encoder-name qwen \
        --seedxunsplash-path data/Seedx_Unsplash \
        --seedxunsplash-prob 1 \
        --qwenvl-image-size 512 \
        --adapter-name dinov2 \
        --adapter-size base \
        --use-adapter True \
        --use-code True \
        --use-fusion True \
        --no-compile \
        
        