DIR_DATA="YOU_ROOT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_impute.py \
    --port 11993\
    --dataset polymnist \
    --unimodal-datapaths-train "$DIR_DATA/MMNIST/train" \
    --unimodal-datapaths-test "$DIR_DATA/MMNIST/test" \
    --observed_mask data/latent_missing_polymnist_0.4.pk \
    --test_missing_ratio 0.4 \
    --target_dir="$PWD/runs/polymnist" \
    --impute_method collaborate \
    --class_dim=512 \
    --beta_content=0.4 \
    --beta_style=10 \
    --likelihood_mnist laplace \
    --batch_size=256 \
    --initial_learning_rate=0.001 \
    --eval_freq=25 \
    --end_epoch=500 \
    --class_per_user 6 \
    --client_num 5 \
    --k_single 1 \
    --dir_clf "$DIR_DATA/pretrained_clfs" \
    --seed 1 \
    --save_figure \
    --communication_freq 1 \
    --validation_freq 5 \
    --use_clf \
    --eval_lr 