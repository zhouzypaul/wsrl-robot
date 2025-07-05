export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python train_wsrl.py "$@" \
    --exp_name=peg_insertion \
    --save_path=logs/wsrl-calql-0519-b256-REDQ-resnet-small-seed2 \
    --eval_checkpoint_step=9000 \
    --eval_n_trajs=20 \
$@  # take in additional arguments
