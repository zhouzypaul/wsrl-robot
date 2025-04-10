export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python visualize_q.py "$@" \
    --exp_name=peg_insertion \
    --calql_checkpoint_path=logs/calql_alpha_5_4_layer-long \
    --data_path=logs/0403/buffer \
    --description="CalQL visualization" \
$@  # take in additional arguments
