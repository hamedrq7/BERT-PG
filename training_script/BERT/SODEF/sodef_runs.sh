# The thing that ran phase1
# python run_sodef.py --output_dir '../phase1testing' --exp_name 'phase1_default_params' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_epoch 10 --phase1_lr 1e-2 --phase1_optim_eps 1e-3 --phase2_epoch 0 --phase2_batch_size 128 --phase3_epochs 0 --seed 100

# python run_sodef.py --output_dir '../dummy' --exp_name 'dummy' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --skip_phase1 --skip_phase2 --seed 100 --no_wandb --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz' --phase3_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/r1=10._r2=1.0_r3=0.1_decay=on_optim=on/phase3/phase3_best_acc_ckpt.pth'

# Dir of adv features: /mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz

torchrun run_sodef.py  --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase1testing/phase1/phase1_best_acc_ckpt.pth' --seed 100 --phase2_batch_size 64 --phase2_numm 64 --phase2_epoch 10 --phase2_weight_diag 10. --phase2_weight_off_diag 1.0 --phase2_weight_f 0.1 --decay_lr --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad --ode_dim 64 --exp_name 'ODE64_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --output_dir '../phase2paramfinding/ODE64_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz'

torchrun run_sodef.py   --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase1testing/phase1/phase1_best_acc_ckpt.pth' --seed 100 --phase2_batch_size 64 --phase2_numm 64 --phase2_epoch 10 --phase2_weight_diag 10. --phase2_weight_off_diag 1.0 --phase2_weight_f 0.1 --decay_lr --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad --ode_dim 128 --exp_name 'ODE128_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --output_dir '../phase2paramfinding/ODE128_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz'


# ### Phase 3 shit: 

# ################################ No freeze
# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=1
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=1
# ODE_EPS=1e-6
# FC_LR=1e-6
# FC_EPS=1e-4
# EXP_NAME="BRIDGE_${BRIDGE}_${BRIDGE_LR}-ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --no_phase3_freeze_bridge_layer --phase3_lr_bridge_layer ${BRIDGE_LR} --phase3_eps_bridge_layer ${BRIDGE_EPS} \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-5
# BRIDGE=1
# BRIDGE_EPS=1e-4
# ODE_LR=1e-3
# ODE=1
# ODE_EPS=1e-5
# FC_LR=1e-5
# FC_EPS=1e-3
# EXP_NAME="BRIDGE_${BRIDGE}_${BRIDGE_LR}-ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --no_phase3_freeze_bridge_layer --phase3_lr_bridge_layer ${BRIDGE_LR} --phase3_eps_bridge_layer ${BRIDGE_EPS} \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-4
# BRIDGE=1
# BRIDGE_EPS=1e-3
# ODE_LR=1e-2
# ODE=1
# ODE_EPS=1e-4
# FC_LR=1e-4
# FC_EPS=1e-2
# EXP_NAME="BRIDGE_${BRIDGE}_${BRIDGE_LR}-ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --no_phase3_freeze_bridge_layer --phase3_lr_bridge_layer ${BRIDGE_LR} --phase3_eps_bridge_layer ${BRIDGE_EPS} \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# ################################ freeze bridge
# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=1
# ODE_EPS=1e-6
# FC_LR=1e-6
# FC_EPS=1e-4
# EXP_NAME="ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-4
# ODE=1
# ODE_EPS=1e-5
# FC_LR=1e-5
# FC_EPS=1e-3
# EXP_NAME="ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-3
# ODE=1
# ODE_EPS=1e-4
# FC_LR=1e-4
# FC_EPS=1e-3
# EXP_NAME="ODE_${ODE}_${ODE_LR}_FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_lr_ode_block ${ODE_LR} --phase3_eps_ode_block ${ODE_EPS} \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 



# ################################ freeze ODE
# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=0
# ODE_EPS=1e-6
# FC_LR=1e-6
# FC_EPS=1e-4
# EXP_NAME="FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_freeze_ode_block \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=0
# ODE_EPS=1e-6
# FC_LR=1e-5
# FC_EPS=1e-3
# EXP_NAME="FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_freeze_ode_block \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 


# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=0
# ODE_EPS=1e-6
# FC_LR=1e-4
# FC_EPS=1e-4
# EXP_NAME="FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_freeze_ode_block \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# PHASE2_EXP_NAME="r1=10._r2=1.0_r3=0.1_decay=on_optim=on"
# PHASE2_MODEL_PATH="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/${PHASE2_EXP_NAME}/phase2/phase2_last_ckpt.pth"
# BRIDGE_LR=1e-6
# BRIDGE=0
# BRIDGE_EPS=1e-4
# ODE_LR=1e-5
# ODE=0
# ODE_EPS=1e-6
# FC_LR=1e-3
# FC_EPS=1e-8
# EXP_NAME="FC_${FC_LR}"
# OUTPUT_DIR="../phase3paramfinding/${EXP_NAME}"
 
# torchrun run_sodef.py \
#     --feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz" \
#     --skip_phase1 \
#     --phase2_model_path $PHASE2_MODEL_PATH \
#     --seed 100 \
#     --exp_name $EXP_NAME \
#     --output_dir $OUTPUT_DIR \
#     --adv_glue_feature_set_dir "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz" \
#     --phase3_freeze_ode_block \
#     --phase3_lr_fc ${FC_LR} --phase3_eps_fc_block ${FC_EPS} 

# torchrun run_sodef.py --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase1testing/phase1/phase1_best_acc_ckpt.pth' --seed 100 --phase2_batch_size 64 --phase2_numm 64 --phase2_epoch 10 --phase2_weight_diag 10. --phase2_weight_off_diag 1.0 --phase2_weight_f 0.1 --decay_lr --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad --ode_dim 512 --exp_name 'ODE512_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --output_dir '../phase2paramfinding/ODE512_r1=10._r2=1.0_r3=0.1_decay=on_optim=on' --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz'

#!/bin/bash

# # ----------- CONSTANT ARGUMENTS -----------
# FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz"
# PHASE1_MODEL="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase1testing/phase1/phase1_best_acc_ckpt.pth"
# ADV_GULE_DOR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz"

# FIXED_ARGS="--feature_set_dir $FEATURE_DIR \
#             --phase1_model_path $PHASE1_MODEL \
#             --adv_glue_feature_set_dir $ADV_GULE_DOR \
#             --seed 100 \
#             --phase2_batch_size 64 \
#             --phase2_numm 64 \
#             --phase2_epoch 10" 

# # ----------- EXPERIMENT VALUE LISTS -----------
# reg1_list=(10. )
# reg2_list=(1.0) # 0.01 0.1 0.0001 
# reg3_list=(0.1) #  0.01

# # Two binary toggles:
# decay_options=("on") #  "off"
# optimizer_options=("on") # "on"
# no_prevs=("on")
# # ----------- LOOP OVER ALL COMBINATIONS -----------
# for r1 in "${reg1_list[@]}"; do
#     for r2 in "${reg2_list[@]}"; do
#         for r3 in "${reg3_list[@]}"; do
#             for decay in "${decay_options[@]}"; do
#                 for optim in "${optimizer_options[@]}"; do
#                     for no_prev in "${no_prevs[@]}"; do

#                         # ---------------- Construct optional args ----------------
#                         decay_arg=""
#                         optim_args=""
#                         no_prev_args=""

#                         if [[ "$decay" == "on" ]]; then
#                             decay_arg="--decay_lr"
#                         fi

#                         if [[ "$optim" == "on" ]]; then
#                             optim_args="--phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
#                         fi

#                         if [[ "$optim" == "sgd" ]]; then
#                             optim_args=" --phase2_optim SGD --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
#                         fi
                        
#                         if [[ "$no_prev" == "on" ]]; then
#                             no_prev_args=" --no_phase2_use_fc_from_phase1 --no_phase3_use_fc_from_phase2"
#                         fi 

#                         # ---------------- Construct experiment name ----------------
#                         exp_name="r1=${r1}_r2=${r2}_r3=${r3}_decay=${decay}_optim=${optim}_prev${no_prev}"

#                         # ---------------- Output directory ----------------
#                         output_dir="../phase2paramfinding/${exp_name}"

#                         # ---------------- Command to run ----------------
#                         CMD="python run_sodef.py \
#                             $FIXED_ARGS \
#                             --phase2_weight_diag $r1 \
#                             --phase2_weight_off_diag $r2 \
#                             --phase2_weight_f $r3 \
#                             $decay_arg \
#                             $optim_args \
#                             $no_prev_args \
#                             --exp_name $exp_name \
#                             --output_dir $output_dir" \

#                         echo "Running: $exp_name"
#                         echo "$CMD"
#                         eval $CMD
#                     done
#                 done
#             done
#         done
#     done
# done

