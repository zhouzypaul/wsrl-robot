export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python train_wsrl.py "$@" \
    --exp_name=peg_insertion \
    --save_path=logs/wsrl-warmup-10000 \
    --pretrained_checkpoint_path=/home/paulzhou/dev-serl-ft/examples/logs/calql_alpha_5_4_layer-reward-1/checkpoint_150000 \
    --actor \
$@  # take in additional arguments
