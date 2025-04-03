export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_rlpd.py "$@" \
    --exp_name=peg_insertion \
    --checkpoint_path=logs/0403 \
    --demo_path=demo_data/peg_insertion_20_demos_2025-04-03_12-12-07.pkl \
    --learner \
