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


# EXPS_NAME="Phase3-FREEZE_FC_baseA_topol=on_lossC=1.0_T=10.0"
# python run_sodef.py  \
#     --skip_phase1 \
#     --phase2_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEF-FreezeFC_lossC/FREEZE_FC_baseA_topol=on_lossC=10.0/phase2/phase2_last_ckpt.pth" \
#     --phase3_freeze_ode_block \
#     --phase3_optim 'SGD' \
#     --phase3_lr_fc 0.0 \
#     --phase3_eps_fc_block 1e-8 \
#     --phase3_epochs 1 \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../DeBERTaSODEF-FreezeFC_lossC/${EXPS_NAME} \
#     --bert_feature_dim 1024 \
#     --use_topol_ode \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
#     --wandb_project_name 'DeBERTa_SODEF_NEW' \
#     --phase3_integration_time 10.0 \

# wandb="DebertASODEF-Phase1"
# EXP_NAME="AdamDefault"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_epoch 20 \
#     --phase1_lr 0.001 \
#     --phase1_optim_eps 1e-8 \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb

# ######################### Vis
# wandb="DebertASODEF-Phase1"
# EXP_NAME="AdamDefault"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEFPhase1/${EXP_NAME}/phase1/phase1_best_acc_ckpt.pth" \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb
# EXP_NAME="Default"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEFPhase1/${EXP_NAME}/phase1/phase1_best_acc_ckpt.pth" \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb
# EXP_NAME="SGDDefault"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEFPhase1/${EXP_NAME}/phase1/phase1_best_acc_ckpt.pth" \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb
# EXP_NAME="SGDLong"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_model_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEFPhase1/${EXP_NAME}/phase1/phase1_best_acc_ckpt.pth" \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb
# ############################### 

# wandb="DebertASODEF-Phase1"
# EXP_NAME="AdamDefault-no-freeze"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_epoch 20 \
#     --phase1_lr 0.001 \
#     --phase1_optim_eps 1e-8 \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb \
#     --no_phase1_freeze_fc

# Center Loss
wandb="DebertASODEF-Phase1"
EXP_NAME="AdamDefault-centerLoss"
python run_sodef.py  \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --phase1_epoch 20 \
    --phase1_lr 0.001 \
    --phase1_optim_eps 1e-8 \
    --skip_phase2 \
    --skip_phase3 \
    --exp_name ${EXP_NAME} \
    --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
    --wandb_project_name $wandb \

#######################################
# EXP_NAME="Default"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_epoch 20 \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --wandb_project_name $wandb

# EXP_NAME="SGDDefault"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_epoch 20 \
#     --phase1_lr 0.001 \
#     --phase1_optim_eps 1e-8 \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --phase1_optim 'SGD' \
#     --wandb_project_name $wandb

# EXP_NAME="SGDLong"
# python run_sodef.py  \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --phase1_epoch 20 \
#     --phase1_lr 0.00005 \
#     --phase1_optim_eps 1e-8 \
#     --skip_phase2 \
#     --skip_phase3 \
#     --exp_name ${EXP_NAME} \
#     --output_dir "../DeBERTaSODEFPhase1/${EXP_NAME}" \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz'\
#     --phase1_optim 'SGD' \
#     --wandb_project_name $wandb

