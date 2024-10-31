# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=12345 \
autoregressive/train/train_c2i.py "$@"
