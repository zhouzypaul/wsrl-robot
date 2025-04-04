export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_calql.py "$@" \
    --exp_name=peg_insertion \
    --calql_checkpoint_path=logs/calql-alpha-1 \
    --data_path=logs/0403/buffer \
$@  # take in additional arguments
