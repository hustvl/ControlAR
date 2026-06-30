#!/bin/bash

torchrun --nnodes=1 --nproc-per-node=4 --master-port=12345 \
        autoregressive/train/train_qwenvl_gpt_edit.py \
        --output-dir ControlAR/outputs \
        --vq-ckpt model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt \
        --qwenvl-weight model_weight/ControlAR/pretrained_models/Qwen3-VL-2B-Instruct \
        --image-size 512 \
        --gpt-model GPT-XL \
        --gpt-ckpt model_weight/ControlAR/pretrained_models/t2i_XL_stage2_512.pt \
        --num-workers 16 \
        --epochs 5 \
        --dataset-list pipe seedxunsplash \
        --pipe-path data/PIPE \
        --pipe-prob 0.5 \
        --encoder-name qwen \
        --qwen-name 3 \
        --seedxunsplash-path data/Seedx_Unsplash \
        --seedxunsplash-prob 1 \
        --qwenvl-image-size 512 \
        --no-compile \
        --adapter-name dinov2 \
        --adapter-size small \
        --use-adapter True \
        --cls-token-num 120 \
        --t5-feature-max-len 120 \
        --use-fusion True \
        --use-code True \
        


