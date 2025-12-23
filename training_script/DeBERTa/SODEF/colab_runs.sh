EXPS_NAME="baseA+expf=75.0"
base_dir="/content/drive/MyDrive/Colab Notebooks/DeBERTa-sst2"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model_path "${base_dir}/phase2_last_ckpt.pth" \
    --phase3_model_path "${base_dir}/phase3_best_acc_ckpt.pth" \
    --exp_name ${EXPS_NAME} \
    --output_dir ../${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --train_feature_set_dir "${base_dir}/train_features.npz" \
    --test_feature_set_dir "${base_dir}/test_features.npz" \
    --seed 100 \
    --adv_glue_feature_set_dir "${base_dir}/advglue_features.npz" \
    --no_wandb \
    --denoising_analysis \
    # --eigval_analysis \