DIR_DATA="YOU_ROOT_DIR"
DIR_EXPERIMENT="$PWD/runs/eicu"  # NOTE: experiment logs are written here

CUDA_VISIBLE_DEVICES=0 python main_impute.py \
    --dataset muse_eicu \
    --target_dir="$DIR_EXPERIMENT" \
    --observed_mask data/latent_missing_muse_eicu_0.16.pk \
    --test_missing_ratio 0.16 \
    --impute_method collaborate \
    --class_dim=256 \
    --beta_content=0.1 \
    --dir_data "$DIR_DATA/muse_eicu" \
    --batch_size=256 \
    --initial_learning_rate=0.0005 \
    --eval_freq=10 \
    --end_epoch=50 \
    --client_num 10 \
    --k_single 0.05 \
    --seed 1 \
    --communication_freq 1 \
    --validation_freq 5 \
    --eval_lr \
    --calc_auprc 
