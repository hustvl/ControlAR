# !/bin/bash
# sleep 36000
# set -x
# export TOKENIZERS_PARALLELISM=true
torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12346 \
autoregressive/train/train_t2i_Janus.py \
--dataset t2i_control \
--image-size 384 \
--adapter-size base \
--data-folder /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_ADE20K/train \
--data-folder2 /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_COCOStuff/train \
--data-folder3 /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/MultiGen20M/train \
--condition-type all \
"$@"
# --data-folder /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/MultiGen20M/train \
# --data-folder /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_ADE20K/train \
# --data-folder2 /data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/Captioned_COCOStuff/train \
# --no-compile \