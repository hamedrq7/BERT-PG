
#!/usr/bin/env bash

# ----------- CONSTANT ARGUMENTS -----------

TR_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats/train_features.npz"
TE_FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats/test-m_features.npz" # test-mm_features.npz
ADV_GULE_DOR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats/adv_glue-m_features.npz" # adv_glue-mm_features.npz
CUDA_ID=0
project_name="MNLI_DeBERTa_SODEF"
# phase1_model_path="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaSODEFPhase1/AdamDefault-saving-best-adv-model/phase1/phase1_best_adv_glue_best_acc_ckpt.pth"
FIXED_ARGS="--train_feature_set_dir $TR_FEATURE_DIR \
            --test_feature_set_dir $TE_FEATURE_DIR \
            --adv_glue_feature_set_dir $ADV_GULE_DOR \
            --seed 111 \
            --phase2_epoch 40 \
            --bert_feature_dim 1024 \
            --phase3_freeze_ode_block \
            --phase3_epochs 15 \
            --phase3_integration_time 8 \
            --wandb_project_name $project_name \
            --cuda_id $CUDA_ID"

# ----------- EXPERIMENT PARAMETER SETS -----------

# (r1, r2, r3)
# base A "10.0 1.0 0.1"
# base B "10.0 10.0 0.2"
reg_sets=(
  "10.0 1.0 0.1"
  "10.0 10.0 0.2"
)

# "1.0 1.0 1.0"
#   "0.1 1.0 1.0"
#   "1.0 0.1 1.0"
#   "1.0 1.0 0.1"
#   "0.1 0.1 1.0"
#   "0.1 1.0 0.1"
#   "1.0 0.1 0.1"
#   "0.01 1.0 1.0"
#   "1.0 0.01 1.0"
#   "1.0 1.0 0.01"
#   "0.01 0.01 1.0"
#   "0.01 1.0 0.01"
#   "1.0 0.01 0.01"
#   "0.01 0.1 1.0"
#   "0.01 1.0 0.1"
#   "0.1 0.01 1.0"
#   "1.0 0.01 0.1"
#   "0.1 1.0 0.01"
#   "1.0 0.1 0.01"
#   "1.0 1.0 0.0"
#   "0.1 1.0 0.0"
#   "1.0 0.1 0.0"
#   "1.0 0.0 1.0"
#   "1.0 0.0 0.1"
#   "0.1 0.0 1.0"
#   "0.0 1.0 1.0"
#   "0.0 0.1 1.0"
#   "0.0 1.0 0.1"

# phase2_exponent, 1.0
# phase2_exponent_off, 0.1 
# phase2_exponent_f, 50.0
# phase2_time_df, 1.0
# phase2_trans, 1.0
# phase2_trans_off_diag, 1.0 
# phase2_integration_time, 5.0
phase2_param_sets=(
  "1.0 0.1 50.0 1.0 1.0 1.0 5.0" 
)
#   "1.0 1.0 50.0 1.0 1.0 1.0 5.0" 
#   "1.0 0.1 50.0 1.0 1.0 1.0 8.0"   
#   "1.0 0.1 10.0 1.0 1.0 1.0 5.0" 
#   "1.0 0.1 75.0 1.0 1.0 1.0 5.0" 
#   "1.0 1.0 50.0 1.0 1.0 1.0 5.0" 
#   "1.0 5.0 50.0 1.0 1.0 1.0 5.0" 
#   "0.1 0.1 50.0 1.0 1.0 1.0 5.0" 
#   "5.0 0.1 50.0 1.0 1.0 1.0 5.0" 
#   "1.0 0.1 50.0 1.0 0.1 1.0 5.0" 
#   "1.0 0.1 50.0 1.0 1.0 0.1 5.0" 

# base
# integ time
# expf
# expf
# expoff
# expoff
# exp
# exp
# phase2_trans
# phase2_trans_off_diag

# ----------- TOGGLES -----------

decay_options=("on")      # "on" or "off"
default_adam=("on")       # "on" or "sgd"
no_prevs=("off")          # "on" or "off"
topol_ode="on"
lossC_set=(0.0 1.0 2.5) # 0.05 0.25 1.0
bs_set=(128) 
# ----------- LOOP OVER ALL EXPERIMENTS -----------

for bs in "${bs_set[@]}"; do
    for reg_set in "${reg_sets[@]}"; do
        read r1 r2 r3 <<< "$reg_set"
        for phase_set in "${phase2_param_sets[@]}"; do
            read exp exp_off exp_f time_df trans trans_off integ_t <<< "$phase_set"
            for lossC in "${lossC_set[@]}"; do
                for decay in "${decay_options[@]}"; do
                    for optim in "${default_adam[@]}"; do
                        for no_prev in "${no_prevs[@]}"; do

                            # ---------------- Construct optional args ----------------
                            decay_arg=""
                            optim_args=""
                            no_prev_args=""
                            ode_args=""

                            if [[ "$topol_ode" == "on" ]]; then
                                ode_args="--use_topol_ode"
                            fi
                            
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
                            exp_name="topol_ode=${topol_ode}_r1=${r1}_r2=${r2}_r3=${r3}_lossC=${lossC}"
                            # exp_name="FREEZE_FC_baseA_topol=${topol_ode}_lossC=${lossC}"

                            # ---------------- Output directory ----------------
                            output_dir="../MNLI-SODEF/matched/${exp_name}"

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
                                --phase2_ce_weight $lossC \
                                --phase2_batch_size $bs \
                                --phase2_numm $bs \
                                $decay_arg \
                                $optim_args \
                                $no_prev_args \
                                $ode_args \
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
done
