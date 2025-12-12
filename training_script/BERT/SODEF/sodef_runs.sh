# The thing that ran phase1
# python run_sodef.py --output_dir '../phase1testing' --exp_name 'phase1_default_params' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_epoch 10 --phase1_lr 1e-2 --phase1_optim_eps 1e-3 --phase2_epoch 0 --phase2_batch_size 128 --phase3_epochs 0 --seed 100

# python run_sodef.py --output_dir '../dummy' --exp_name 'dummy' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --skip_phase1 --skip_phase2 --seed 100 --no_wandb --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz' --phase3_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding/r1=10._r2=1.0_r3=0.1_decay=on_optim=on/phase3/phase3_best_acc_ckpt.pth'

# Dir of adv features: /mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz

#!/bin/bash

# ----------- CONSTANT ARGUMENTS -----------
FEATURE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz"
PHASE1_MODEL="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase1testing/phase1/phase1_best_acc_ckpt.pth"
ADV_GULE_DOR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/eval/base_model-adv_glue/AdvGLUE_val_feats.npz"

FIXED_ARGS="--feature_set_dir $FEATURE_DIR \
            --phase1_model_path $PHASE1_MODEL \
            --adv_glue_feature_set_dir $ADV_GULE_DOR \
            --seed 100 \
            --phase2_batch_size 64 \
            --phase2_numm 64 \
            --phase2_epoch 10" 

# ----------- EXPERIMENT VALUE LISTS -----------
reg1_list=(10. )
reg2_list=(1.0) # 0.01 0.1 0.0001 
reg3_list=(0.1) #  0.01

# Two binary toggles:
decay_options=("on") #  "off"
optimizer_options=("on") # "on"
no_prevs=("on")
# ----------- LOOP OVER ALL COMBINATIONS -----------
for r1 in "${reg1_list[@]}"; do
    for r2 in "${reg2_list[@]}"; do
        for r3 in "${reg3_list[@]}"; do
            for decay in "${decay_options[@]}"; do
                for optim in "${optimizer_options[@]}"; do
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
                        exp_name="r1=${r1}_r2=${r2}_r3=${r3}_decay=${decay}_optim=${optim}_prev${no_prev}"

                        # ---------------- Output directory ----------------
                        output_dir="../phase2paramfinding/${exp_name}"

                        # ---------------- Command to run ----------------
                        CMD="python run_sodef.py \
                            $FIXED_ARGS \
                            --phase2_weight_diag $r1 \
                            --phase2_weight_off_diag $r2 \
                            --phase2_weight_f $r3 \
                            $decay_arg \
                            $optim_args \
                            $no_prev \
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

