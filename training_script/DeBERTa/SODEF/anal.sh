EXPS_NAME="Phase3-baseA-topolSodef"
python run_sodef.py  \
    --skip_phase1 \
    --skip_phase2 \
    --phase3_model "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/DeBERTaFirstSODEF-TopolSodef/r1=10.0_r2=1.0_r3=0.1_exp=1.0_expoff_0.1_expf=50.0_timedf_1.0_trans_1.0_transoff_1.0_T=5.0_dafaultAdam=on_noprev=off_topol_ode=on/phase3/phase3_best_acc_ckpt.pth" \
    --exp_name ${EXPS_NAME} \
    --output_dir ../DeBERTaSodefPhase3Tuning/${EXPS_NAME} \
    --bert_feature_dim 1024 \
    --use_topol_ode \
    --train_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/train_features.npz' \
    --test_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/test_features.npz' \
    --seed 100 \
    --adv_glue_feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2/feats/advglue_features.npz' \
    --no_wandb \
    --eigval_analysis \

