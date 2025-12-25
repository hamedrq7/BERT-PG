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
import json

def main():

    # python run_sodef.py --output_dir '../sodef_testing' --exp_name 'first_test' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2//saving_feats/0_feats.npz' --phase1_epochs 2 --phase2_epoch 1 --phase2_batch_size 128 --phase3_epochs 2
    # python run_sodef.py --output_dir '../sodef_testing' --exp_name 'second_test' --feature_set_dir '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2//saving_feats/0_feats.npz' --phase1_epochs 2 --phase2_epoch 1 --phase2_batch_size 128 --phase3_epochs 2 --seed 100
    
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    """
    with open("args.json", "r") as f:
        args_dict = json.load(f)
    """

    if args.wandb:
        wandb.init(project=f"{args.wandb_project_name}", config=vars(args), name=args.exp_name)

    print("Experiment:", args.exp_name)
    print("Output dir:", args.output_dir)
    
    if ((not torch.cuda.is_available()) or (not args.use_cuda)):
        device = 'cpu' 
    else:
        torch.cuda.set_device(args.cuda_id)
        device = torch.device(f'cuda:{args.cuda_id}')
        
    from general_utils import set_seed_reproducability
    set_seed_reproducability(args.seed)

    # TODOOOOOOOOOO Set seed
    # TODOOOOOOOOOO bert sanity check 
    advglue_feature_loader = None
    if args.adv_glue_feature_set_dir is not None:
        ds = get_adv_glue_feature_dataset(args.adv_glue_feature_set_dir, True)
        advglue_feature_loader = DataLoader(
            ds,
            batch_size=128,
            shuffle=True, num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )

    if not args.skip_phase1: 
        from train_utils import train_phase1, load_phase1
        phase1_model = train_phase1(args, device, adv_glue_loader=advglue_feature_loader) if args.phase1_model_path is None else load_phase1(args, device, True)
        # base + phase1/phase1_best_acc_ckpt.pth
        from analysis_utils import tsne_plot_phase1
        # tsne_plot_phase1(args, phase1_model, device, advglue_loader=advglue_feature_loader)

    phase2_model = None
    if not args.skip_phase2: 
        from train_utils import train_phase2, load_phase2
        phase2_model = train_phase2(phase1_model, args, device, adv_glue_loader=advglue_feature_loader) if args.phase2_model_path is None else load_phase2(args, device, False, advglue_feature_loader)
        # base + phase2/phase2_last_ckpt.pth
    
        print('******* After phase2... ******* ')
        from train_utils import test_phase2
        test_phase2(phase2_model, args, device, advglue_feature_loader)

        from analysis_utils import tsne_plot_phase1
        tsne_plot_phase1(args, phase2_model, device, 'phase2', advglue_feature_loader)

    print('Starting phase3...')
    phase3_model = None
    if not args.skip_phase3: 
        from train_utils import train_phase3, load_phase3
        phase3_model = load_phase3(args, device, False)  if args.phase3_model_path is not None else train_phase3(phase2_model, args, device, adv_glue_loader=advglue_feature_loader) 
        # base + phase3/phase3_best_acc_ckpt.pth

        from train_utils import test_adv_glue
        res = test_adv_glue(args, device, phase3_model, advglue_feature_loader)
        if args.wandb: 
            wandb.log({'AdvGLUE': res['acc']})
    
    if args.eigval_analysis: 
        from analysis_utils import eigval_analysis
        from train_utils import get_feature_dataloader
        models = [['phase3', phase3_model]]
        if phase2_model is not None: 
            models.append(['phase2', phase2_model])
        
        for model in models: 
            print('Analysing ', model[0], ' model')
            trainloader, testloader = get_feature_dataloader(args, args.phase3_batch_size)
            
            eigval_analysis(model[1], trainloader, device,
                            output_path=f'{args.output_dir}/{model[0]}', 
                            phase='train', num_points=1000) 

            eigval_analysis(model[1], testloader, device,
                            output_path=f'{args.output_dir}/{model[0]}', 
                            phase='test', num_points=100) 

            if advglue_feature_loader is not None:             
                eigval_analysis(model[1], advglue_feature_loader, device,
                                output_path=f'{args.output_dir}/{model[0]}', 
                                phase='advglue', num_points=147) 
    
    if args.denoising_analysis: 
        from analysis_utils import denoising_analysis
        from train_utils import get_feature_dataloader
        trainloader, testloader = get_feature_dataloader(args, args.phase3_batch_size)
        denoising_analysis(phase2_model, phase3_model, trainloader, testloader, device, advglue_feature_loader) 

        from analysis_utils import raddddddi, raddddddi_rand
        # raddddddi(phase3_model, 'train_phase3', trainloader, device)
        # raddddddi(phase2_model, 'train_phase2', trainloader, device)
        raddddddi_rand(phase3_model, 'test_phase3', testloader, device)
        raddddddi_rand(phase3_model, 'adv_phase3', advglue_feature_loader, device)
        raddddddi_rand(phase3_model, 'train_phase3', trainloader, device)

        raddddddi_rand(phase2_model, 'test_phase2', testloader, device)
        raddddddi_rand(phase2_model, 'adv_phase2', advglue_feature_loader, device)
        raddddddi_rand(phase2_model, 'train_phase2', trainloader, device)
        

    if args.wandb:
        wandb.finish()
    
if __name__ == "__main__":
    main()