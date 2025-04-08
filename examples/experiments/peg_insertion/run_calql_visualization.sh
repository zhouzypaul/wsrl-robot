export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python visualize_q.py "$@" \
    --exp_name=peg_insertion \
    --calql_checkpoint_path=logs/calql_alpha_5_4_layer-long \
    --data_path=demo_data/peg_insertion_20_demos_2025-02-20_15-34-54.pkl \
    --description="CalQL visualization" \
$@  # take in additional arguments
