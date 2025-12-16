# Full 3 phases
# python run_sodef.py  --bert_feature_dim 1024 --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' --seed 100 --phase1_epoch 10 --phase1_lr 1e-2 --phase1_optim_eps 1e-3 --phase2_batch_size 64 --phase2_numm 64 --phase2_epoch 10 --phase2_weight_diag 10. --phase2_weight_off_diag 1.0 --phase2_weight_f 0.1 --decay_lr --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad --exp_name 'ODE64_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --output_dir '../DeBERTaFirstSODEF/ODE64_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' --phase3_freeze_ode_block --phase3_lr_fc 1e-5 --phase3_eps_fc_block 1e-3


#### Tuning phase1: 
# default
# PHASE1_EPOCHS=4
# PHASE1_OPTIM='ADAM'
# PHASE1_LR=1e-1
# PHASE1_EPS=1e-2
# EXPS_NAME="Phase1-Tuning/eps_${PHASE1_EPS}-optim_${PHASE1_OPTIM}-lr_${PHASE1_LR}-eps_${PHASE1_EPS}"
# python run_sodef.py  \
#     --phase1_epoch ${PHASE1_EPOCHS} \
#     --phase1_optim ${PHASE1_OPTIM} \
#     --phase1_lr ${PHASE1_LR} \
#     --phase1_optim_eps ${PHASE1_EPS} \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../DeBERTaFirstSODEF/${EXPS_NAME} \
#     --phase2_epoch 0 \
#     --phase3_epochs 0 \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \

# # Adam default
# PHASE1_EPOCHS=10
# PHASE1_OPTIM='ADAM'
# PHASE1_LR=1e-3
# PHASE1_EPS=1e-8
# EXPS_NAME="Phase1-Tuning/eps_${PHASE1_EPS}-optim_${PHASE1_OPTIM}-lr_${PHASE1_LR}-eps_${PHASE1_EPS}"
# python run_sodef.py  \
#     --phase1_epoch ${PHASE1_EPOCHS} \
#     --phase1_optim ${PHASE1_OPTIM} \
#     --phase1_lr ${PHASE1_LR} \
#     --phase1_optim_eps ${PHASE1_EPS} \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../DeBERTaFirstSODEF/${EXPS_NAME} \
#     --phase2_epoch 0 \
#     --phase3_epochs 0 \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \

# SGD default
PHASE1_EPOCHS=10
PHASE1_OPTIM='SGD'
PHASE1_LR=1e-3
EXPS_NAME="Phase1-Tuning/eps_${PHASE1_EPS}-optim_${PHASE1_OPTIM}-lr_${PHASE1_LR}-eps_${PHASE1_EPS}"
python run_sodef.py  \
    --phase1_epoch ${PHASE1_EPOCHS} \
    --phase1_optim ${PHASE1_OPTIM} \
    --phase1_lr ${PHASE1_LR} \
    --phase1_optim_eps ${PHASE1_EPS} \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaFirstSODEF/${EXPS_NAME} \
    --phase2_epoch 0 \
    --phase3_epochs 0 \
    --bert_feature_dim 1024 \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
