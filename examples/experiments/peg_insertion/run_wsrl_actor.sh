export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python train_wsrl.py "$@" \
    --exp_name=peg_insertion \
    --save_path=logs/wsrl-0519 \
    --pretrained_checkpoint_path=logs/calql-0513-b256-d098-resnet-large-jitter0/checkpoint_200000 \
    --actor \
$@  # take in additional arguments
