# EXPS_NAME="baseA+expf=75.0"
# base_dir="/content/drive/MyDrive/Colab Notebooks/DeBERTa-sst2"
# python run_sodef.py  \
#     --skip_phase1 \
#     --phase2_model_path "${base_dir}/phase2_last_ckpt.pth" \
#     --phase3_model_path "${base_dir}/phase3_best_acc_ckpt.pth" \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../${EXPS_NAME} \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir "${base_dir}/train_features.npz" \
#     --test_feature_set_dir "${base_dir}/test_features.npz" \
#     --seed 100 \
#     --adv_glue_feature_set_dir "${base_dir}/advglue_features.npz" \
#     --no_wandb \
#     --denoising_analysis \
#     # --eigval_analysis \


EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=on_lossC=1.0_T=10.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=on_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --use_topol_ode \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 10.0 \

EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=on_lossC=1.0_T=5.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=on_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --use_topol_ode \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 5.0 \

EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=on_lossC=1.0_T=1.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=on_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --use_topol_ode \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 1.0 \


EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=off_lossC=10.0_T=10.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=off_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 10.0 \

EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=off_lossC=10.0_T=5.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=off_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 5.0 \

EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=off_lossC=10.0_T=1.0"
python run_sodef.py  \
    --skip_phase1 \
    --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=off_lossC=10.0/phase2/phase2_last_ckpt.pth" \
    --phase3_freeze_ode_block \
    --phase3_optim 'SGD' \
    --phase3_lr_fc 0.0 \
    --phase3_eps_fc_block 1e-8 \
    --phase3_epochs 1 \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --wandb_project_name 'DeBERTa_SODEF_NEW' \
    --phase3_integration_time 1.0 \