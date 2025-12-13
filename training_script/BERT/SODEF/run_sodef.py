import torch
import torch.nn as nn 
import numpy as np 
from tqdm import trange
from torch.utils.data import DataLoader
import sys 
import os 

BERT_CKPT_DIR = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2'
FEATS_DIR = f'{BERT_CKPT_DIR}/saving_feats/{0}_feats.npz'
CLF_LAYER_DIR = f'{BERT_CKPT_DIR}/bert_clf.pth'

# def bert_fc_features_sanity_check(bert_clf_layer, trainloader, testloader, device): 
#     tr_res = test_ce(-1, bert_clf_layer, trainloader, device, nn.CrossEntropyLoss(), 110, '', save_name='bert_fc_sanity_check_train')
#     te_res = test_ce(-1, bert_clf_layer, testloader, device, nn.CrossEntropyLoss(), 110, '', save_name='bert_fc_sanity_check_test')

#     print('Train Acc, Loss', tr_res['acc'], tr_res['loss'])
#     print('Test Acc, Loss', te_res['acc'], te_res['loss'])

from general_utils import get_args
from data_utils import get_adv_glue_feature_dataset

import wandb

def main():

    # python run_sodef.py --output_dir '../sodef_testing' --exp_name 'first_test' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2//saving_feats/0_feats.npz' --phase1_epochs 2 --phase2_epoch 1 --phase2_batch_size 128 --phase3_epochs 2
    # python run_sodef.py --output_dir '../sodef_testing' --exp_name 'second_test' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2//saving_feats/0_feats.npz' --phase1_epochs 2 --phase2_epoch 1 --phase2_batch_size 128 --phase3_epochs 2 --seed 100
    
    args = get_args()

    if args.wandb:
        wandb.init(project="BERT-SODEF-phase3-HYPERPARAM-FINDING", config=vars(args), name=args.exp_name)

    print("Experiment:", args.exp_name)
    print("Output dir:", args.output_dir)
    
    device = 'cpu' if ((not torch.cuda.is_available()) or (not args.use_cuda)) else torch.device('cuda:0')


    from general_utils import set_seed_reproducability
    set_seed_reproducability(args.seed)

    # TODOOOOOOOOOO Set seed
    # TODOOOOOOOOOO bert sanity check 
    AdvGLUE=False
    if args.adv_glue_feature_set_dir is not None:
        ds = get_adv_glue_feature_dataset(args.adv_glue_feature_set_dir)
        advglue_feature_loader = DataLoader(
            ds,
            batch_size=128,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        AdvGLUE=True

    if not args.skip_phase1: 
        from train_utils import train_phase1, load_phase1
        phase1_model = train_phase1(args, device) if args.phase1_model_path is None else load_phase1(args, device, True)
        # base + phase1/phase1_best_acc_ckpt.pth

    if not args.skip_phase2: 
        from train_utils import train_phase2, load_phase2
        phase2_model = train_phase2(phase1_model, args, device) if args.phase2_model_path is None else load_phase2(args, device, True)
        # base + phase2/phase2_last_ckpt.pth
    
    from train_utils import train_phase3, load_phase3
    phase3_model = load_phase3(args, device, True)  if args.phase3_model_path is not None else train_phase3(phase2_model, args, device) 
    # base + phase3/phase3_best_acc_ckpt.pth

    
    if AdvGLUE:
        from train_utils import test_adv_glue
        res = test_adv_glue(args, device, phase3_model, advglue_feature_loader)
        if args.wandb: 
            wandb.log({'AdvGLUE': res['acc']})
    
    if args.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()