for i in {19..19}; do
    ITER=$((i * 10000))
    CKPT_NAME=$(printf "%07d.pt" $ITER)
    CKPT_PATH="ControlAR/outputs/ControlAR-2025-12-14-07-21-44/checkpoints/$CKPT_NAME"
    torchrun \
    --nnodes=1 --nproc_per_node=4 --node_rank=0 \
    --master_port=12349 \
    autoregressive/test/test_t2i_Janus.py \
    --gpt-ckpt "$CKPT_PATH" \
    --use-adapter True \
    --image-size 384 \
    --data-folder data/MultiGen20M/val \
    --condition-type depth \
    --sample-dir sample/depth \
    --temperature 0.7 \
    --cfg-scale 5 \
    "$@"

    echo "Depth for iteration $CKPT_PATH: "
    python evaluations/depth_rmse.py

done 