# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_port=12346 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
"$@"
