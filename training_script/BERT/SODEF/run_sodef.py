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

def main():
    # python run_sodef.py --output_dir '../sodef_testing' --exp_name 'first_test' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2//saving_feats/{0}_feats.npz' --phase1_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/phase1/phase1_best_acc_ckpt.pth' --phase2_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/phase2model_9.pth' --phase3_model_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/phase3_best_acc_ckpt.pth'
    args = get_args()

    print("Experiment:", args.exp_name)
    print("Output dir:", args.output_dir)
    
    device = 'cpu' if ((not torch.cuda.is_available()) or (not args.use_cuda)) else torch.device('cuda:0')

    # Set seed
    # bert sanity check 

    from train_utils import train_phase1, load_phase1
    phase1_model = train_phase1(args, device) if args.phase1_model_path is None else load_phase1(args, device, True)
    # base + phase1/phase1_best_acc_ckpt.pth

    from train_utils import train_phase2, load_phase2
    phase2_model = train_phase2(phase1_model, args, device) if args.phase2_model_path is None else load_phase2(args, device, True)
    # base + phase2/phase2_last_ckpt.pth
    
    from train_utils import train_phase3, load_phase3
    phase3_model = train_phase3(phase2_model, args, device) if args.phase3_model_path is None else load_phase3(args, device, True)
    # base + phase3/phase3_best_acc_ckpt.pth

    
    
if __name__ == "__main__":
    main()