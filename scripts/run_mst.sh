DIR_DATA="YOU_ROOT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_impute.py \
    --port 11993\
    --dataset mnistsvhntext \
    --dir_data "$DIR_DATA/mnistsvhntext" \
    --observed_mask data/latent_missing_mnistsvhntext_0.33.pk \
    --target_dir="$PWD/runs/mst" \
    --test_missing_ratio 0.33 \
    --impute_method collaborate \
    --teacher_forcing=0.2 \
    --class_dim=256 \
    --beta_content=0.4 \
    --beta_style=10 \
    --dim=64 \
    --batch_size=256 \
    --initial_learning_rate=0.001 \
    --eval_freq=25 \
    --end_epoch=300 \
    --class_per_user 10 \
    --client_num 5 \
    --k_single 1 \
    --dir_clf "$DIR_DATA/pretrained_clfs" \
    --save_figure \
    --communication_freq 1 \
    --validation_freq 5 \
    --seed 1 \
    --use_clf \
    --eval_lr \