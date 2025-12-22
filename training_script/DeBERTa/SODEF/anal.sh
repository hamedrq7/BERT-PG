# EXP_NAME="r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on"
# python run_sodef.py  \
#     --skip_phase1 \
#     --phase2_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-TopolSodef/${EXP_NAME}/phase2/phase2_last_ckpt.pth" \
#     --phase3_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-TopolSodef/${EXP_NAME}/phase3/phase3_best_acc_ckpt.pth" \
#     --exp_name ${EXP_NAME} \
#     --output_dir ../DeBERTaFirstSODEF-TopolSodef/${EXP_NAME} \
#     --bert_feature_dim 1024 \
#     --use_topol_ode \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
#     --no_wandb \
#     --eigval_analysis \

# Analysing  phase3  model
# DS:  torch.Size([67349, 1024]) torch.Size([67349])
# DS:  torch.Size([872, 1024]) torch.Size([872])
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 527/527 [00:08<00:00, 60.07it/s]

# === Exp D summary - train ===

# [SODEF]
#   num_points: 1000
#   num_eigs_total: 64000
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.04677039384841919
#   max_real_median: 0.04378331080079079
#   max_real_p95: 0.06778045743703842
#   max_real_max: 0.16782322525978088
#   real_mean: -0.5109157562255859
#   real_min: -1.8190314769744873
#   real_max: 0.16782322525978088
#   imag_abs_mean: 0.0043759532272815704
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase3/eigval_train.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 36.25it/s]

# === Exp D summary - test ===

# [SODEF]
#   num_points: 100
#   num_eigs_total: 6400
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.05006434768438339
#   max_real_median: 0.05086016654968262
#   max_real_p95: 0.10280990600585938
#   max_real_max: 0.14176326990127563
#   real_mean: -0.5264106392860413
#   real_min: -1.81375253200531
#   real_max: 0.14176326990127563
#   imag_abs_mean: 0.004506791941821575
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase3/eigval_test.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 16.47it/s]

# === Exp D summary - advglue ===

# [SODEF]
#   num_points: 147
#   num_eigs_total: 9408
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.10240921378135681
#   max_real_median: 0.10230488330125809
#   max_real_p95: 0.16477297246456146
#   max_real_max: 0.1701667606830597
#   real_mean: -0.5442652106285095
#   real_min: -1.8155982494354248
#   real_max: 0.1701667606830597
#   imag_abs_mean: 0.005603841971606016
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase3/eigval_advglue.png
# Analysing  phase2  model
# DS:  torch.Size([67349, 1024]) torch.Size([67349])
# DS:  torch.Size([872, 1024]) torch.Size([872])
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 527/527 [00:01<00:00, 317.55it/s]

# === Exp D summary - train ===

# [SODEF]
#   num_points: 1000
#   num_eigs_total: 64000
#   frac_Re_gt_0: 0.0004531250160653144
#   frac_Re_gt_1e-6: 0.0004531250160653144
#   max_real_mean: -0.08515802770853043
#   max_real_median: -0.07115311920642853
#   max_real_p95: -0.0318542905151844
#   max_real_max: 0.09978684037923813
#   real_mean: -0.5444446802139282
#   real_min: -1.8454406261444092
#   real_max: 0.09978684037923813
#   imag_abs_mean: 0.0022705255541950464
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase2/eigval_train.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 57.27it/s]

# === Exp D summary - test ===

# [SODEF]
#   num_points: 100
#   num_eigs_total: 6400
#   frac_Re_gt_0: 0.0017187499906867743
#   frac_Re_gt_1e-6: 0.0017187499906867743
#   max_real_mean: -0.08353962004184723
#   max_real_median: -0.07405909895896912
#   max_real_p95: 0.0688595250248909
#   max_real_max: 0.08608929067850113
#   real_mean: -0.5709435343742371
#   real_min: -1.8641753196716309
#   real_max: 0.08608929067850113
#   imag_abs_mean: 0.0024235278833657503
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase2/eigval_test.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 19.64it/s]

# === Exp D summary - advglue ===

# [SODEF]
#   num_points: 147
#   num_eigs_total: 9408
#   frac_Re_gt_0: 0.006802720949053764
#   frac_Re_gt_1e-6: 0.006802720949053764
#   max_real_mean: -0.008964186534285545
#   max_real_median: -0.01622939109802246
#   max_real_p95: 0.08967199921607971
#   max_real_max: 0.09823577851057053
#   real_mean: -0.5664491057395935
#   real_min: -1.8640385866165161
#   real_max: 0.09823577851057053
#   imag_abs_mean: 0.002332789823412895
# [saved] ../DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase2/eigval_advglue.png


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


# === Exp D summary - train ===

# [SODEF]
#   num_points: 1000
#   num_eigs_total: 64000
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.05440841242671013
#   max_real_median: 0.05445341020822525
#   max_real_p95: 0.0553334578871727
#   max_real_max: 0.055705875158309937
#   real_mean: -0.5127473473548889
#   real_min: -0.9292629361152649
#   real_max: 0.055705875158309937
#   imag_abs_mean: 0.00025031124823726714
# [saved] ../DeBERTaFirstSODEF-Phase2/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off/phase3/eigval_train.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 40.67it/s]

# === Exp D summary - test ===

# [SODEF]
#   num_points: 100
#   num_eigs_total: 6400
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.05490604415535927
#   max_real_median: 0.05529598891735077
#   max_real_p95: 0.05561034753918648
#   max_real_max: 0.05576383322477341
#   real_mean: -0.5173353552818298
#   real_min: -0.9300227761268616
#   real_max: 0.05576383322477341
#   imag_abs_mean: 0.00011344554513925686
# [saved] ../DeBERTaFirstSODEF-Phase2/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off/phase3/eigval_test.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 15.12it/s]

# === Exp D summary - advglue ===

# [SODEF]
#   num_points: 147
#   num_eigs_total: 9408
#   frac_Re_gt_0: 0.015625
#   frac_Re_gt_1e-6: 0.015625
#   max_real_mean: 0.055318791419267654
#   max_real_median: 0.055480413138866425
#   max_real_p95: 0.05575811490416527
#   max_real_max: 0.055767714977264404
#   real_mean: -0.5201013088226318
#   real_min: -0.9301224946975708
#   real_max: 0.055767714977264404
#   imag_abs_mean: 0.00012473358947318047
# [saved] ../DeBERTaFirstSODEF-Phase2/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off/phase3/eigval_advglue.png
# Analysing  phase2  model
# DS:  torch.Size([67349, 1024]) torch.Size([67349])
# DS:  torch.Size([872, 1024]) torch.Size([872])
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 527/527 [00:01<00:00, 339.01it/s]

# === Exp D summary - train ===

# [SODEF]
#   num_points: 1000
#   num_eigs_total: 64000
#   frac_Re_gt_0: 0.0
#   frac_Re_gt_1e-6: 0.0
#   max_real_mean: -0.0005909347091801465
#   max_real_median: -0.0005913132335990667
#   max_real_p95: -0.0005865544080734253
#   max_real_max: -0.0005792927695438266
#   real_mean: -0.5235969424247742
#   real_min: -0.9333597421646118
#   real_max: -0.0005792927695438266
#   imag_abs_mean: 6.142141501186416e-05
# [saved] ../DeBERTaFirstSODEF-Phase2/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off/phase2/eigval_train.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 50.06it/s]

# === Exp D summary - test ===

# [SODEF]
#   num_points: 100
#   num_eigs_total: 6400
#   frac_Re_gt_0: 0.0
#   frac_Re_gt_1e-6: 0.0
#   max_real_mean: -0.0005953893414698541
#   max_real_median: -0.0005955533124506474
#   max_real_p95: -0.0005943914875388145
#   max_real_max: -0.0005911741172894835
#   real_mean: -0.5277595520019531
#   real_min: -0.9351462721824646
#   real_max: -0.0005911741172894835
#   imag_abs_mean: 3.6385346902534366e-05
# [saved] ../DeBERTaFirstSODEF-Phase2/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off/phase2/eigval_test.png
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 17.54it/s]

# === Exp D summary - advglue ===

# [SODEF]
#   num_points: 147
#   num_eigs_total: 9408
#   frac_Re_gt_0: 0.0
#   frac_Re_gt_1e-6: 0.0
#   max_real_mean: -0.0005946307210251689
#   max_real_median: -0.00059500802308321
#   max_real_p95: -0.0005923512508161366
#   max_real_max: -0.0005888465093448758
#   real_mean: -0.5269289612770081
#   real_min: -0.9351468086242676
#   real_max: -0.0005888465093448758
#   imag_abs_mean: 5.5773823987692595e-05




EXPS_NAME="r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=75.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-Phase2/${EXPS_NAME}/phase2/phase2_last_ckpt.pth" \
    --phase3_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-Phase2/${EXPS_NAME}/phase3/phase3_best_acc_ckpt.pth" \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaFirstSODEF-Phase2/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --no_wandb \
    --eigval_analysis \