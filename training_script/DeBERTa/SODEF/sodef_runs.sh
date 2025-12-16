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

# # SGD default
# PHASE1_EPOCHS=20
# PHASE1_OPTIM='SGD'
# PHASE1_LR=1e-3
# EXPS_NAME="Phase1-Tuning/eps_${PHASE1_EPS}-optim_${PHASE1_OPTIM}-lr_${PHASE1_LR}-eps_${PHASE1_EPS}"
# python run_sodef.py  \
#     --phase1_epoch ${PHASE1_EPOCHS} \
#     --phase1_optim ${PHASE1_OPTIM} \
#     --phase1_lr ${PHASE1_LR} \
#     --exp_name ${EXPS_NAME} \
#     --output_dir ../DeBERTaFirstSODEF/${EXPS_NAME} \
#     --phase2_epoch 0 \
#     --phase3_epochs 0 \
#     --bert_feature_dim 1024 \
#     --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
#     --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
#     --seed 100 \
#     --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \


# Running Phase2: 
/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF/Phase1-Tuning/eps_-optim_SGD-lr_1e-3-eps_/phase1/phase1_best_acc_ckpt.pth


#!/usr/bin/env bash

# ----------- CONSTANT ARGUMENTS -----------

TR_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz"
TE_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz"
PHASE1_MODEL="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF/Phase1-Tuning/eps_-optim_SGD-lr_1e-3-eps_/phase1/phase1_best_acc_ckpt.pth"
ADV_GULE_DOR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz"

FIXED_ARGS="--train_feature_set_dir $TR_FEATURE_DIR \
            --test_feature_set_dir $TE_FEATURE_DIR \
            --phase1_model_path $PHASE1_MODEL \
            --adv_glue_feature_set_dir $ADV_GULE_DOR \
            --seed 100 \
            --phase2_batch_size 64 \
            --phase2_numm 64 \
            --phase2_epoch 5 \
            --bert_feature_dim 1024"

# ----------- EXPERIMENT PARAMETER SETS -----------

# (r1, r2, r3)
reg_sets=(
  "10.0 1.0 0.1", 
  "10.0 10.0 0.2"
)

# phase2_exponent, 1.0
# phase2_exponent_off, 0.1 
# phase2_exponent_f, 50.0
# phase2_time_df, 1.0
# phase2_trans, 1.0
# phase2_trans_off_diag, 1.0 
# phase2_integration_time, 5.0
phase2_param_sets=(
  "1.0 0.1 50.0 1.0 1.0 1.0 5.0" # base
  "1.0 0.1 50.0 1.0 1.0 1.0 8.0" # integ time 
  "1.0 0.1 10.0 1.0 1.0 1.0 5.0" # expf
  "1.0 0.1 75.0 1.0 1.0 1.0 5.0" # expf
  "1.0 1.0 50.0 1.0 1.0 1.0 5.0" # expoff
  "1.0 5.0 50.0 1.0 1.0 1.0 5.0" # expoff
  "0.1 0.1 50.0 1.0 1.0 1.0 5.0" # exp
  "5.0 0.1 50.0 1.0 1.0 1.0 5.0" # exp
  "1.0 0.1 50.0 1.0 0.1 1.0 5.0" # phase2_trans
  "1.0 0.1 50.0 1.0 1.0 0.1 5.0" # phase2_trans_off_diag
)

# ----------- TOGGLES -----------

decay_options=("on")      # "on" or "off"
default_adam=("on")       # "on" or "sgd"
no_prevs=("off")          # "on" or "off"

# ----------- LOOP OVER ALL EXPERIMENTS -----------

for reg_set in "${reg_sets[@]}"; do
    read r1 r2 r3 <<< "$reg_set"

    for phase_set in "${phase2_param_sets[@]}"; do
        read exp exp_off exp_f time_df trans trans_off integ_t <<< "$phase_set"
        
        for decay in "${decay_options[@]}"; do
            for optim in "${default_adam[@]}"; do
                for no_prev in "${no_prevs[@]}"; do

                    # ---------------- Construct optional args ----------------
                    decay_arg=""
                    optim_args=""
                    no_prev_args=""

                    if [[ "$decay" == "on" ]]; then
                        decay_arg="--decay_lr"
                    fi

                    if [[ "$optim" == "on" ]]; then
                        optim_args="--phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
                    fi

                    if [[ "$optim" == "sgd" ]]; then
                        optim_args="--phase2_optim SGD --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
                    fi

                    if [[ "$no_prev" == "on" ]]; then
                        no_prev_args="--no_phase2_use_fc_from_phase1 --no_phase3_use_fc_from_phase2"
                    fi

                    # ---------------- Construct experiment name ----------------
                    exp_name="r1=${r1}_r2=${r2}_r3=${r3}_exp=${exp}_expoff_${exp_off}_expf=${exp_f}_timedf_${time_df}_trans_${trans}_transoff_${trans_off}_T=${integ_t}_dafaultAdam=${optim}_noprev=${no_prev}"

                    # ---------------- Output directory ----------------
                    output_dir="../DeBERTaFirstSODEF-Phase2/${exp_name}"

                    # ---------------- Command to run ----------------
                    CMD="python run_sodef.py \
                        $FIXED_ARGS \
                        --phase2_weight_diag $r1 \
                        --phase2_weight_off_diag $r2 \
                        --phase2_weight_f $r3 \
                        --phase2_exponent $exp \
                        --phase2_exponent_off $exp_off \
                        --phase2_exponent_f $exp_f \
                        --phase2_time_df $time_df \
                        --phase2_trans $trans \
                        --phase2_trans_off_diag $trans_off \
                        --phase2_integration_time $integ_t \
                        $decay_arg \
                        $optim_args \
                        $no_prev_args \
                        --exp_name $exp_name \
                        --output_dir $output_dir"

                    echo "Running: $exp_name"
                    echo "$CMD"
                    eval $CMD

                done
            done
        done
    done
done


# ----------- CONSTANT ARGUMENTS -----------

TR_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz"
TE_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz"
PHASE1_MODEL="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF/Phase1-Tuning/eps_-optim_SGD-lr_1e-3-eps_/phase1/phase1_best_acc_ckpt.pth"
ADV_GULE_DOR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz"

FIXED_ARGS="--train_feature_set_dir $TR_FEATURE_DIR \
            --test_feature_set_dir $TE_FEATURE_DIR \
            --phase1_model_path $PHASE1_MODEL \
            --adv_glue_feature_set_dir $ADV_GULE_DOR \
            --seed 100 \
            --phase2_batch_size 64 \
            --phase2_numm 64 \
            --phase2_epoch 5 \
            --bert_feature_dim 1024" 

# ----------- EXPERIMENT VALUE LISTS -----------
reg1_list=(10. )
reg2_list=(1.0) # 0.01 0.1 0.0001 
reg3_list=(0.1) #  0.01

phase2_exponent=1.0
phase2_exponent_off=0.1
phase2_exponent_f=50.
phase2_time_df=1.
phase2_trans=1.
phase2_trans_off_diag=1.
phase2_integration_time=5.

# Two binary toggles:
decay_options=("on") #  "off"
default_adam=("on") 
no_prevs=("off")
# ----------- LOOP OVER ALL COMBINATIONS -----------
for r1 in "${reg1_list[@]}"; do
    for r2 in "${reg2_list[@]}"; do
        for r3 in "${reg3_list[@]}"; do
            for decay in "${decay_options[@]}"; do
                for optim in "${default_adam[@]}"; do
                    for no_prev in "${no_prevs[@]}"; do

                        # ---------------- Construct optional args ----------------
                        decay_arg=""
                        optim_args=""
                        no_prev_args=""

                        if [[ "$decay" == "on" ]]; then
                            decay_arg="--decay_lr"
                        fi

                        if [[ "$optim" == "on" ]]; then
                            optim_args="--phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
                        fi

                        if [[ "$optim" == "sgd" ]]; then
                            optim_args=" --phase2_optim SGD --phase2_lr 0.001 --phase2_eps 1e-08 --no_phase2_amsgrad"
                        fi
                        
                        if [[ "$no_prev" == "on" ]]; then
                            no_prev_args=" --no_phase2_use_fc_from_phase1 --no_phase3_use_fc_from_phase2"
                        fi 

                        # ---------------- Construct experiment name ----------------
                        exp_name="r1=${r1}_r2=${r2}_r3=${r3}_default_adam=${optim}_no_prev${no_prev}"

                        # ---------------- Output directory ----------------
                        output_dir="../DeBERTaFirstSODEF-Phase2/${exp_name}"

                        # ---------------- Command to run ----------------
                        CMD="python run_sodef.py \
                            $FIXED_ARGS \
                            --phase2_weight_diag $r1 \
                            --phase2_weight_off_diag $r2 \
                            --phase2_weight_f $r3 \
                            $decay_arg \
                            $optim_args \
                            $no_prev_args \
                            --exp_name $exp_name \
                            --output_dir $output_dir" \

                        echo "Running: $exp_name"
                        echo "$CMD"
                        eval $CMD
                    done
                done
            done
        done
    done
done

