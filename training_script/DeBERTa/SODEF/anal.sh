EXPS_NAME="r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-TopolSodef/${EXP_NAME}/phase2/phase2_last_ckpt.pth" \
    --phase3_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-TopolSodef/${EXP_NAME}/phase3/phase3_best_acc_ckpt.pth" \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaFirstSODEF-TopolSodef/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --use_topol_ode \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --no_wandb \
    --eigval_analysis \

# [SODEF]
#   num_points: 1000
#   num_eigs_total: 64000
#   frac_Re_gt_0: 0.01560937613248825
#   frac_Re_gt_1e-6: 0.01560937613248825
#   max_real_mean: 0.04774120822548866
#   max_real_median: 0.044024884700775146
#   max_real_p95: 0.07009299099445343
#   max_real_max: 0.16955794394016266
#   real_mean: -0.5113537907600403
#   real_min: -1.8317242860794067
#   real_max: 0.16955794394016266
#   imag_abs_mean: 0.00442200293764472

# EXPS_NAME="r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off"
# python run_sodef.py  \
#     --skip_phase1 \
#     --phase2_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-Phase2/${EXPS_NAME}/phase2/phase2_last_ckpt.pth" \
#     --phase3_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-Phase2/${EXPS_NAME}/phase3/phase3_best_acc_ckpt.pth" \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../DeBERTaFirstSODEF-Phase2/${EXPS_NAME} \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
#     --no_wandb \
#     --eigval_analysis \
