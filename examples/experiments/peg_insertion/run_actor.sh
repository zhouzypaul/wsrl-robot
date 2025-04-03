export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python train_rlpd.py "$@" \
    --exp_name=peg_insertion \
    --checkpoint_path=logs/0403 \
    --actor \
$@  # take in additional arguments
