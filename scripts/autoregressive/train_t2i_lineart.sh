# !/bin/bash
# sleep 36000
set -x
export TOKENIZERS_PARALLELISM=true
torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12346 \
autoregressive/train/train_t2i_lineart.py \
--vq-ckpt checkpoints/vq/vq_ds16_t2i.pt \
--gpt-ckpt checkpoints/llamagen/t2i_XL_stage2_512.pt \
--dataset t2i_control \
--image-size 512 \
--cloud-save-path output \
--code-path data/MultiGen20M/train \
"$@"
