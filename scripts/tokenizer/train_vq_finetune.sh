# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12345 \
tokenizer/tokenizer_image/vq_train.py \
--finetune \
--disc-start 0 \
--vq-ckpt vq_ds16_c2i.pt \
--dataset imagenet_code \
--cloud-save-path output/cloud_disk \
"$@"

