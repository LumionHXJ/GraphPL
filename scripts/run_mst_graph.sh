DIR_DATA="YOU_ROOT_DIR"
DIR_EXPERIMENT="$PWD/runs/mst"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main_impute.py \
    --port 11993\
    --dataset mnistsvhntext \
    --dir_data "$DIR_DATA/mnistsvhntext" \
    --observed_mask data/latent_missing_mnistsvhntext_0.33.pk \
    --target_dir="$DIR_EXPERIMENT" \
    --test_missing_ratio 0.33 \
    --impute_method graph \
    --teacher_forcing=0.2 \
    --class_dim=128 \
    --beta_content=0.4 \
    --beta_style=10 \
    --gnn_layers=3 \
    --dim=64 \
    --batch_size=256 \
    --initial_learning_rate=0.001 \
    --eval_freq=25 \
    --end_epoch=300 \
    --fusion_method zero \
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