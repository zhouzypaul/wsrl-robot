export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_calql.py "$@" \
    --exp_name=peg_insertion \
    --calql_checkpoint_path=logs/calql-alpha-1 \
    --data_path=logs/rlpd-0430-b256-discount-0.98/buffer \
$@  # take in additional arguments
