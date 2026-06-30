#!/bin/bash

torchrun --nnodes=1 --nproc-per-node=4 --master-port=12345 \
        autoregressive/train/train_edit.py \
        --output-dir /data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/outputs \
        --vq-ckpt /data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/vq_ds16_t2i.pt \
        --image-size 512 \
        --gpt-model GPT-XL \
        --gpt-ckpt /data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/pretrained_models/t2i_XL_stage2_512.pt \
        --num-workers 16 \
        --epochs 5 \
        --dataset-list pipe seedxunsplash\
        --pipe-path /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/PIPE \
        --pipe-prob 0.5 \
        --encoder-name vqgan \
        --seedxunsplash-path /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Seedx_Unsplash \
        --seedxunsplash-prob 1