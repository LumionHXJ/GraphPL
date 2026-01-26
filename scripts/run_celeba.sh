DIR_DATA="YOU_ROOT_DIR"
DIR_EXPERIMENT="$PWD/runs/celeba"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_impute.py \
    --dataset celeba \
    --port 12591 \
    --dir_data "$DIR_DATA/CelebA" \
    --observed_mask data/latent_missing_celeba_0.25.pk \
    --target_dir="$DIR_EXPERIMENT" \
    --test_missing_ratio 0.25 \
    --impute_method collaborate \
    --class_dim=512 \
    --beta_style=10 \
    --beta_content=1.0 \
    --batch_size=128 \
    --initial_learning_rate=5e-4 \
    --eval_freq=10 \
    --end_epoch=100 \
    --dir_clf "$DIR_DATA/pretrained_clfs" \
    --client_num 5 \
    --k_single 1 \
    --seed 1 \
    --communication_freq 1 \
    --validation_freq 5 \
    --save_figure \
    --use_clf \
    --eval_lr 