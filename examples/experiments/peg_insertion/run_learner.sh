export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_rlpd.py "$@" \
    --exp_name=peg_insertion \
    --checkpoint_path=logs/0430 \
    --demo_path=logs/rlpd-0512-b256-d098-large/buffer \
    --learner \
$@  # take in additional arguments
