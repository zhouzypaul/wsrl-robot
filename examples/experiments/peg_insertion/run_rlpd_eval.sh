export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_rlpd.py "$@" \
    --exp_name=peg_insertion \
    --checkpoint_path=logs/rlpd-0519-b256-REDQ-resnet-small-seed3 \
    --eval_checkpoint_step=9000 \
    --eval_n_trajs=20 \
    --actor \
$@  # take in additional arguments
