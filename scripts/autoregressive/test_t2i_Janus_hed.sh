for i in {11..11}; do
    ITER=$((i * 10000))
    CKPT_NAME=$(printf "%07d.pt" $ITER)
    CKPT_PATH="ControlAR/outputs/ControlAR-2025-12-14-07-18-26/checkpoints/$CKPT_NAME"
    torchrun \
    --nnodes=1 --nproc_per_node=4 --node_rank=0 \
    --master_port=12349 \
    autoregressive/test/test_t2i_Janus.py \
    --gpt-ckpt "$CKPT_PATH" \
    --use-adapter True \
    --image-size 384 \
    --data-folder data/MultiGen20M/val \
    --condition-type hed \
    --sample-dir sample/hed \
    --temperature 0.7 \
    --cfg-scale 2.5 \
    "$@"

    echo "Hed for iteration $CKPT_PATH: "
    python evaluations/hed_ssim.py 

done