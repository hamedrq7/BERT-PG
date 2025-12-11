# # LOADING MODELS AND DATA
# parser.add_argument("--output_dir", type=str, required=True)
# parser.add_argument("--feature_set_dir", type=str, default=None) #  --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz'
# parser.add_argument("--exp_name", type=str, required=True)


# # ---------------------------
# # PHASE 2
# # ---------------------------

# # TRAINING METHOD
# parser.add_argument("--no_phase2_use_fc_from_phase1", action="store_false", dest="phase2_use_fc_from_phase1",
#     help="Disable phase2_use_fc_from_phase1 (default: enabled)")

# parser.add_argument("--phase2_batch_size", type=int, default=32)
# parser.add_argument("--phase2_epoch", type=int, default=20)

# # HYPERPARAMS
# parser.add_argument("--phase2_weight_diag", type=float, default=10) # reg1
# parser.add_argument("--phase2_weight_off_diag", type=float, default=0.) # reg2
# parser.add_argument("--phase2_weight_f", type=float, default=0.1) # reg3
# parser.add_argument("--phase2_weight_norm", type=float, default=0.)
# parser.add_argument("--phase2_weight_lossc", type=float, default=0.)
# parser.add_argument("--phase2_exponent", type=float, default=1.0)
# parser.add_argument("--phase2_exponent_off", type=float, default=0.1)
# parser.add_argument("--phase2_exponent_f", type=float, default=50)
# parser.add_argument("--phase2_time_df", type=float, default=1.)
# parser.add_argument("--phase2_trans", type=float, default=1.0)
# parser.add_argument("--phase2_trans_off_diag", type=float, default=1.0)
# parser.add_argument("--phase2_numm", type=int, default=16)
# parser.add_argument("--phase2_integration_time", type=float, default=5.)

# # OPTIM PARAMS
# parser.add_argument("--phase2_optim", type=str, default="ADAM")
# parser.add_argument("--phase2_lr", type=float, default=1e-2)
# parser.add_argument("--phase2_eps", type=float, default=1e-3)
# parser.add_argument("--no_phase2_amsgrad", action="store_false", dest="phase2_amsgrad",
#     help="Disable phase2_amsgrad (default: enabled)")

# # ---------------------------
# # PHASE 3
# # ---------------------------
# parser.add_argument("--no_phase3_use_fc_from_phase2", action="store_false", dest="phase3_use_fc_from_phase2",
#     help="Disable phase3_use_fc_from_phase2 (default: enabled)")

# # HYPERPARAMS
# parser.add_argument("--phase3_optim", type=str, default="ADAM")
# parser.add_argument("--phase3_lr_ode_block", type=float, default=1e-5)
# parser.add_argument("--phase3_eps_ode_block", type=float, default=1e-6)
# parser.add_argument("--phase3_lr_fc", type=float, default=1e-6)
# parser.add_argument("--phase3_eps_fc_block", type=float, default=1e-4)
# parser.add_argument("--no_phase3_amsgrad", action="store_false", dest="phase3_amsgrad",
#     help="Disable phase3_amsgrad (default: enabled)")

# parser.add_argument("--phase3_epochs", type=int, default=10)
# parser.add_argument("--phase3_batch_size", type=int, default=128)

# # LOGGING
# parser.add_argument("--phase3_model_path", type=str, default=None)


python run_sodef.py --output_dir '../phase1testing' --exp_name 'phase1_default_params' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/saving_feats/0_feats.npz' --phase1_epoch 10 --phase2_epoch 0 --phase2_batch_size 128 --phase3_epochs 0 --seed 100