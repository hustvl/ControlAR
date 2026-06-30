for i in {12..12}; do
    ITER=$((i * 10000))
    # 格式化为7位数字，例如 0010000, 0020000, ..., 1000000
    CKPT_NAME=$(printf "%07d.pt" $ITER)
    CKPT_PATH="ControlAR/outputs/ControlAR-2025-12-14-07-19-51/checkpoints/$CKPT_NAME"
    torchrun \
    --nnodes=1 --nproc_per_node=4 --node_rank=0 \
    --master_port=12349 \
    autoregressive/test/test_t2i_Janus.py \
    --gpt-ckpt "$CKPT_PATH" \
    --use-adapter True \
    --image-size 384 \
    --data-folder data/MultiGen20M/val \
    --condition-type lineart \
    --sample-dir sample/lineart \
    --temperature 0.7 \
    --cfg-scale 3
    "$@"

    echo "Lineart for iteration $CKPT_PATH: "
    python evaluations/lineart_ssim.py

done 