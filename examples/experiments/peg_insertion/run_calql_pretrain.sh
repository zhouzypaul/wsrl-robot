export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_calql.py "$@" \
    --exp_name=peg_insertion \
    --calql_checkpoint_path=calql_ckpts \
    --demo_path=logs/full_obs/buffer/transitions_20000.pkl \
