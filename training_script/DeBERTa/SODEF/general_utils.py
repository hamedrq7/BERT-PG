import torch 
import torch.nn as nn 
import numpy as np 
import argparse

def get_loss(loss_name: str):
    if loss_name == 'CE': 
        return nn.CrossEntropyLoss()      
    else: 
        print(f'Loss {loss_name} not implemented')

def get_args():
    parser = argparse.ArgumentParser()

    # ---------------------------
    # GLOBAL PARAMS
    # ---------------------------

    # DATALOADER
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory",
            help="Disable pin_memory (default: enabled)")   
    parser.add_argument("--no_wandb", action="store_false", dest="wandb",
            help="Disable wandb logging (default: enabled)")   
    parser.add_argument("--no_use_cuda", action="store_false", dest="use_cuda",
            help="Disable use_cuda (default: enabled)")   
    parser.add_argument("--cuda_id", type=int, default=0)
    

    # LOADING MODELS AND DATA
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_set_dir", type=str, default=None)
    parser.add_argument("--train_feature_set_dir", type=str, default=None)
    parser.add_argument("--test_feature_set_dir", type=str, default=None)
    parser.add_argument("--bert_dir", type=str, default=None)
    parser.add_argument("--bert_clf_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--adv_glue_feature_set_dir", type=str, default=None)
    
    parser.add_argument("--skip_phase1", action="store_true", help="Skips phase1, make sure to pass mode path for other phases")
    parser.add_argument("--skip_phase2", action="store_true", help="Skips phase2, make sure to pass mode path for phase3")
    # parser.add_argument("--skip_phase3", action="store_true", help="Skips phase1, make sure to pass mode path for other phases")
    
    parser.add_argument("--eigval_analysis", action="store_true", help="do eigval analysis on ode block")
    

    # ARCHITECTURE
    parser.add_argument("--bert_feature_dim", type=int, default=768)
    parser.add_argument("--ode_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--no_ignore_dropout", action="store_false", dest="ignore_dropout",
        help="Disable ignore_dropout (default: enabled)"
    )
    parser.add_argument("--use_topol_ode", action="store_true", help="Use a heavier ODE architecture")
    parser.add_argument("--seed", type=int, default=42)

    # ---------------------------
    # PHASE 1
    # ---------------------------

    # TRAINING PARAMS
    parser.add_argument("--phase1_epochs", type=int, default=4)
    parser.add_argument("--phase1_batch_size", type=int, default=128)
    parser.add_argument("--no_phase1_freeze_backbone", action="store_false", dest="phase1_freeze_backbone",
        help="Disable phase1_freeze_backbone (default: enabled)")
    
    # OPTIM PARAMS
    parser.add_argument("--phase1_optim", type=str, default="ADAM")
    parser.add_argument("--phase1_lr", type=float, default=1e-1)
    parser.add_argument("--phase1_optim_eps", type=float, default=1e-2)
    parser.add_argument("--no_phase1_amsgrad", action="store_false", dest="phase1_amsgrad",
        help="Disable phase1_amsgrad (default: enabled)")
    parser.add_argument("--phase1_loss", type=str, default="CE")

    # LOGGING
    parser.add_argument("--phase1_metric", nargs="+", default=["ACC", "CE_LOSS"])
    parser.add_argument("--phase1_model_path", type=str, default=None)
    parser.add_argument("--phase1_save_path", type=str, default="phase1")

    # ---------------------------
    # PHASE 2
    # ---------------------------

    # TRAINING METHOD
    parser.add_argument("--no_phase2_freeze_backbone", action="store_false", dest="phase2_freeze_backbone",
        help="Disable phase2_freeze_backbone (default: enabled)")
    parser.add_argument("--no_phase2_freeze_bridge_layer", action="store_false", dest="phase2_freeze_bridge_layer",
        help="Disable phase2_freeze_bridge_layer (default: enabled)")
    parser.add_argument("--no_phase2_freeze_fc", action="store_false", dest="phase2_freeze_fc",
        help="Disable phase2_freeze_fc (default: enabled)")
    parser.add_argument("--no_phase2_use_fc_from_phase1", action="store_false", dest="phase2_use_fc_from_phase1",
        help="Disable phase2_use_fc_from_phase1 (default: enabled)")
    # TRAINING PARAMS
    parser.add_argument("--phase2_batch_size", type=int, default=32)
    parser.add_argument("--phase2_epoch", type=int, default=20)

    # HYPERPARAMS
    parser.add_argument("--phase2_weight_diag", type=float, default=10)
    parser.add_argument("--phase2_weight_off_diag", type=float, default=0.)
    parser.add_argument("--phase2_weight_f", type=float, default=0.1)

    parser.add_argument("--phase2_weight_norm", type=float, default=0.)
    parser.add_argument("--phase2_weight_lossc", type=float, default=0.)
    
    parser.add_argument("--phase2_exponent", type=float, default=1.0)
    parser.add_argument("--phase2_exponent_off", type=float, default=0.1)
    parser.add_argument("--phase2_exponent_f", type=float, default=50)
    parser.add_argument("--phase2_time_df", type=float, default=1.)
    parser.add_argument("--phase2_trans", type=float, default=1.0)
    parser.add_argument("--phase2_trans_off_diag", type=float, default=1.0)
    parser.add_argument("--phase2_numm", type=int, default=16)
    parser.add_argument("--phase2_integration_time", type=float, default=5.)
    
    # OPTIM PARAMS
    parser.add_argument("--phase2_optim", type=str, default="ADAM")
    parser.add_argument("--phase2_lr", type=float, default=1e-2)
    parser.add_argument("--decay_lr", action="store_true", help="Decay lr for phase 2 at 3/4 of training")
    parser.add_argument("--phase2_eps", type=float, default=1e-3)
    parser.add_argument("--no_phase2_amsgrad", action="store_false", dest="phase2_amsgrad",
        help="Disable phase2_amsgrad (default: enabled)")

    # LOGGING
    parser.add_argument("--phase2_metric", nargs="+", default=["ALL"])
    parser.add_argument("--phase2_model_path", type=str, default=None)
    parser.add_argument("--phase2_save_path", type=str, default="phase2")

    # ---------------------------
    # PHASE 3
    # ---------------------------

    # TRAINING METHOD    
    parser.add_argument("--no_phase3_freeze_backbone", action="store_false", dest="phase3_freeze_backbone",
        help="Disable phase3_freeze_backbone (default: enabled)")
    parser.add_argument("--no_phase3_freeze_bridge_layer", action="store_false", dest="phase3_freeze_bridge_layer",
        help="Disable phase3_freeze_bridge_layer (default: enabled)")
    parser.add_argument("--phase3_freeze_ode_block", action="store_true", help="phase3_freeze_ode_block (default: enabled)")
    
    
    parser.add_argument("--no_phase3_use_fc_from_phase2", action="store_false", dest="phase3_use_fc_from_phase2",
        help="Disable phase3_use_fc_from_phase2 (default: enabled)")

    # HYPERPARAMS
    parser.add_argument("--phase3_optim", type=str, default="ADAM")
    parser.add_argument("--phase3_lr_bridge_layer", type=float, default=1e-6)
    parser.add_argument("--phase3_eps_bridge_layer", type=float, default=1e-4)
    
    parser.add_argument("--phase3_lr_ode_block", type=float, default=1e-5)
    parser.add_argument("--phase3_eps_ode_block", type=float, default=1e-6)
    parser.add_argument("--phase3_lr_fc", type=float, default=1e-6)
    parser.add_argument("--phase3_eps_fc_block", type=float, default=1e-4)
    parser.add_argument("--no_phase3_amsgrad", action="store_false", dest="phase3_amsgrad",
        help="Disable phase3_amsgrad (default: enabled)")
    parser.add_argument("--phase3_integration_time", type=float, default=5.)

    parser.add_argument("--phase3_loss", type=str, default="CE")
    parser.add_argument("--phase3_epochs", type=int, default=10)
    parser.add_argument("--phase3_batch_size", type=int, default=128)

    # LOGGING
    parser.add_argument("--phase3_metric", nargs="+", default=["ACC"])
    parser.add_argument("--phase3_save_path", type=str, default="phase3")
    parser.add_argument("--phase3_model_path", type=str, default=None)

    args = parser.parse_args()
    
    # Sanitiy checks: 
    if args.train_feature_set_dir is None or args.test_feature_set_dir is None: 
        assert args.feature_set_dir is not None, 'Have not implemented running feature from bert with this script yet, use the `get_feats...sh` script and pass directory of feature set to this script'
    # assert args.bert_feature_dim == 768, 'not sure anything other than bert-base-cased works with this script'
    # assert args.ode_dim == 64, 'not sure other values work'
    assert args.num_classes == 2, 'for more classes, the MAX_ROW_DIS function is not sufficient (Complete this later)'
    assert args.phase1_freeze_backbone, 'Not implemented yet (you need to add bert finetuning for this)'
    assert args.phase2_freeze_backbone, 'Not implemented yet (you need to add bert finetuning for this)'
    assert args.phase3_freeze_backbone, 'Not implemented yet (you need to add bert finetuning for this)'
    if args.skip_phase1:
        if args.skip_phase2:
            assert args.phase3_model_path is not None, 'Pass something'
    
    return args

import random
def set_seed_reproducability(seed): 
    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True  # Enables cudnn
        torch.backends.cudnn.benchmark = True  # It should improve runtime performances when batch shape is fixed. See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results

# def main():
#     args = get_args()
#     """
#     Namespace(num_workers=2, pin_memory=True, use_cuda=True, output_dir='test', feature_set_dir='test', bert_dir=None, bert_clf_dir=None, exp_name='test', bert_feature_dim=768, ode_dim=64, num_classes=2, ignore_dropout=True, seed=42, phase1_epochs=4, phase1_batch_size=128, phase1_freeze_backbone=True, phase1_optim='ADAM', phase1_lr=0.1, phase1_optim_eps=0.01, phase1_amsgrad=True, phase1_loss='CE', phase1_metric=['ACC', 'CE_LOSS'], phase1_model_path=None, phase1_save_path='phase1', phase2_freeze_backbone=True, phase2_freeze_bridge_layer=True, phase2_freeze_fc=True, phase2_use_fc_from_phase1=True, phase2_batch_size=32, phase2_epoch=20, phase2_weight_diag=10, phase2_weight_off_diag=0.0, phase2_weight_f=0.1, phase2_weight_norm=0.0, phase2_weight_lossc=0.0, phase2_exponent=1.0, phase2_exponent_off=0.1, phase2_exponent_f=50, phase2_time_df=1.0, phase2_trans=1.0, phase2_trans_off_diag=1.0, phase2_numm=16, phase2_optim='ADAM', phase2_lr=0.01, phase2_eps=0.001, phase2_amsgrad=True, phase2_metric=['ALL'], phase2_model_path=None, phase2_path='phase2', phase3_freeze_backbone=True, phase3_freeze_bridge_layer=True, phase3_use_fc_from_phase2=True, phase3_optim='ADAM', phase3_lr_ode_block=1e-05, phase3_eps_ode_block=1e-06, phase3_lr_fc=1e-06, phase3_eps_fc_block=0.0001, phase3_amsgrad=True, phase3_loss='CE', 
# phase3_epochs=10, phase3_batch_size=128, phase3_metric=['ACC'], phase3_path='phase3', phase3_model_path=None)
#     """
#     print(args)

    
    
# if __name__ == "__main__":
#     main()