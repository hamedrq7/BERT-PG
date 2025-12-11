import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import geotorch
from torchdiffeq import odeint_adjoint as odeint
import torch 
import math 
from tqdm import trange, tqdm 
import os 
from model_utils import (
    MLP_OUT_ORTH_X_X, 
    get_max_row_dist_for_2_classes, 
    check_max_row_dist_matrix,
    MLP_OUT_BALL_given_mat,
    Phase1Model,
    
)
from general_utils import (
    get_loss
)
from data_utils import get_feature_dataloader
from model_utils import ODEfunc_mlp, Phase2Model, ODEBlocktemp, MLP_OUT_LINEAR
from data_utils import inf_generator
from loss_utils import df_dz_regularizer, f_regularizer
from model_utils import SingleOutputWrapper
from model_utils import ODEBlock, Phase3Model, get_a_phase1_model, get_a_phase2_model, get_a_phase3_model
import wandb

def train_ce_one_epoch(epoch, model, loader, device, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
        outputs = model(x)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        'model': model, 
        'loss': train_loss/(batch_idx+1), 
        'acc': correct/total
    }


def test_ce_one_epoch(epoch, model, loader, device, criterion, best_acc, do_save, save_folder, save_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):            
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = model(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss/(batch_idx+1)
    acc = correct/total

    # Save checkpoint.
    if acc > best_acc:
        best_acc = acc
        if do_save: 
            print(f'Saving at epoch {epoch} with acc {acc} ...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, save_folder+f'/{save_name}_best_acc_ckpt.pth')
            
    return {
        'model': model,
        'loss': avg_loss, 
        'acc': acc,
        'best_acc': best_acc
    }


def train_phase1(args, device): 

    # if you dont ignore dropout, you need to do a drop out at the very begining of every model
    # at the input data essentially 
    assert args.ignore_dropout, 'if you dont ignore dropout, you need to do a drop out at the very begining of every model at the input data essentially' 
    assert args.phase1_loss == 'CE', 'only CE for now'
    assert args.phase1_optim == 'ADAM', 'Only Adam for now'

    trainloader, testloader = get_feature_dataloader(args, args.phase1_batch_size)

    phase1_model = Phase1Model(args.bert_feature_dim, args.ode_dim, args.num_classes).to(device)
    phase1_model.set_all_req_grads()
    print('phase1_model', phase1_model)

    criterion = get_loss(args.phase1_loss)
    optimizer = torch.optim.Adam(phase1_model.parameters(), lr=args.phase1_lr, eps=args.phase1_optim_eps, amsgrad=args.phase1_amsgrad)

    save_path = os.path.join(args.output_dir, args.phase1_save_path)
    os.makedirs(save_path, exist_ok=True)

    best_acc = 0  # best test accuracy

    # args.phase1_metric
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    
    for epoch in trange(0, args.phase1_epochs):
        tr_results = train_ce_one_epoch(
            epoch=epoch, 
            model=phase1_model, 
            loader=trainloader, 
            device=device, 
            optimizer=optimizer,
            criterion=criterion
        )
        te_results = test_ce_one_epoch(
            epoch=epoch, 
            model=phase1_model, 
            loader=testloader,
            device=device,
            criterion=criterion,
            best_acc=best_acc,
            do_save=True,
            save_folder= save_path, # ?
            save_name= 'phase1' # ? 
        )
        best_acc = te_results['best_acc']

        # print('tr_acc, tr_loss', tr_results['acc'], tr_results['loss'])
        # print('te_acc, te_loss', te_results['acc'], te_results['loss'])
        
        train_loss_history.append(tr_results['loss'])
        train_acc_history.append(tr_results['acc'])
        
        test_loss_history.append(te_results['loss'])
        test_acc_history.append(te_results['acc'])

        wandb.log({
            'phase1/step': epoch, 
            'phase1/train_ce': tr_results['loss'], 'phase1/train_acc': tr_results['acc'], 
            'phase1/test_ce': te_results['loss'], 'phase1/test_acc': te_results['acc'], 
        })

    return phase1_model

def load_phase1(args, device, sanity_check = True): 
    saved_temp = torch.load(args.phase1_model_path)
    statedic_temp = saved_temp[list(saved_temp.keys())[0]] # ['model']

    phase1_model = get_a_phase1_model(args.bert_feature_dim, args.ode_dim, args.num_classes)
    phase1_model.load_state_dict(statedic_temp)
    phase1_model = phase1_model.to(device)

    if sanity_check: 
        print('Sanity Check on loaded model phase 1 ... ')
        trainloader, testloader = get_feature_dataloader(args, args.phase1_batch_size)
        criterion = get_loss(args.phase1_loss)
        tr_res = test_ce_one_epoch(-1, phase1_model, trainloader, device, criterion, 110, False, None, None)
        te_res = test_ce_one_epoch(-1, phase1_model, testloader, device, criterion, 110, False, None, None)
        print('Train Acc, Loss', tr_res['acc'], tr_res['loss'])
        print('Test Acc, Loss', te_res['acc'], te_res['loss'])

    return phase1_model

def train_phase2(phase1_model, args, device, ): 
    # # HYPERPARAMS
    # parser.add_argument("--phase2_weight_diag", type=float, default=10)
    # parser.add_argument("--phase2_weight_off_diag", type=float, default=0.)
    # parser.add_argument("--phase2_weight_f", type=float, default=0.1)
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

    # # LOGGING
    # parser.add_argument("--phase2_metric", nargs="+", default=["ALL"])

    # if you dont ignore dropout, you need to do a drop out at the very begining of every model
    # at the input data essentially 
    assert args.ignore_dropout, 'if you dont ignore dropout, you need to do a drop out at the very begining of every model at the input data essentially' 
    assert args.phase2_optim == 'ADAM', 'Only Adam for now'

    save_path = os.path.join(args.output_dir, args.phase2_save_path)
    os.makedirs(save_path, exist_ok=True)

    trainloader, testloader = get_feature_dataloader(args, args.phase2_batch_size)
    
    odefunc = ODEfunc_mlp(args.ode_dim)
    phase2_model = Phase2Model(
        phase1_model.orthogonal_bridge_layer, 
        ODEBlocktemp(odefunc, args.phase2_integration_time), 
        MLP_OUT_LINEAR(dim1=args.ode_dim, dim2=args.num_classes) if args.phase2_use_fc_from_phase1 else phase1_model.fc
    )
    phase2_model.set_all_req_grads(True)
    freeze_layers = []
    if args.phase2_freeze_bridge_layer: 
        freeze_layers.append('bridge_layer')
    if args.phase2_freeze_fc:
        freeze_layers.append('fc')
    phase2_model.freeze_layer_given_name(freeze_layers)
    phase2_model = phase2_model.to(device)

    train_data_gen = inf_generator(trainloader)
    batches_per_epoch = len(trainloader)

    optimizer = torch.optim.Adam(phase2_model.parameters(), lr=args.phase2_lr, eps=args.phase2_eps, amsgrad=args.phase2_amsgrad)
    


    total_iters = args.phase2_epoch * batches_per_epoch
    decay_iter = int(0.75 * total_iters)
    def lr_lambda(step):
        if step >= decay_iter:
            return 0.1
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for itr in trange(args.phase2_epoch * batches_per_epoch):
        phase2_model.train()
        optimizer.zero_grad()
        x, y = train_data_gen.__next__()
        x = x.to(device)

        # modulelist = list(ODE_FCmodel)
        # y0 = x
        # x = modulelist[0](x)
        # y1 = x
        # y00 = y0 #.clone().detach().requires_grad_(True)
        x, y00, _ = phase2_model(x)
        regu1, regu2  = df_dz_regularizer(None, x, numm=args.phase2_numm, odefunc=odefunc, time_df=args.phase2_time_df, exponent=args.phase2_exponent, trans=args.phase2_trans, exponent_off=args.phase2_exponent_off, transoffdig=args.phase2_trans_off_diag, device=device)
        regu1 = regu1.mean()
        regu2 = regu2.mean()
        # print("regu1:weight_diag "+str(regu1.item())+':'+str(args.phase2_weight_diag))
        # print("regu2:weight_offdiag "+str(regu2.item())+':'+str(args.phase2_weight_off_diag))
        regu3 = f_regularizer(None, x, odefunc=odefunc, time_df=args.phase2_time_df, device=device, exponent_f=args.phase2_exponent_f)
        regu3 = regu3.mean()
        # print("regu3:weight_f "+str(regu3.item())+':'+str(args.phase2_weight_f))
        loss = args.phase2_weight_f*regu3 + args.phase2_weight_diag*regu1+ args.phase2_weight_off_diag*regu2
        # print("loss"+str(loss.item()))

        loss.backward()
        optimizer.step()
        if args.decay_lr:
            scheduler.step()
        torch.cuda.empty_cache()
        
        wandb.log({
            'phase2/step': itr,
            'phase2/regu1': regu1.item(),
            'phase2/regu2': regu2.item(),
            'phase2/regu3': regu3.item(),
        })

        if itr % 20 == 0: 
            # print('itr', itr)
            # print("regu1:weight_diag "+str(regu1.item())+':'+str(args.phase2_weight_diag))
            # print("regu2:weight_offdiag "+str(regu2.item())+':'+str(args.phase2_weight_off_diag))
            # print("regu3:weight_f "+str(regu3.item())+':'+str(args.phase2_weight_f))
            # print("loss"+str(loss.item()))
            pass

        
        if itr % batches_per_epoch == 0:
            
            tr_res = test_ce_one_epoch(
                epoch=-1, 
                model=SingleOutputWrapper(phase2_model), 
                loader=trainloader, 
                device=device, 
                criterion=nn.CrossEntropyLoss(), 
                best_acc=110, 
                do_save=False, save_folder=None, save_name=None)
            te_res = test_ce_one_epoch(
                epoch=-1, 
                model=SingleOutputWrapper(phase2_model), 
                loader=testloader, 
                device=device, 
                criterion=nn.CrossEntropyLoss(), 
                best_acc=110, 
                do_save=False, save_folder=None, save_name=None)
            
            print('itr = ', itr, 'Train Acc, Loss', tr_res['acc'], tr_res['loss'])
            print('itr = ', itr, 'Test Acc, Loss', te_res['acc'], te_res['loss'])

            wandb.log({
                'phase2/step': itr,
                'phase2/train_acc': tr_res['acc'],
                'phase2/test_acc': te_res['acc'],
            })

            torch.save(
                {'model': phase2_model.state_dict(), 'itr': itr}, 
                save_path+f'/phase2_last_ckpt.pth'
            )
                
    return phase2_model

def load_phase2(args, device, sanity_check = True): 
    phase2_model = get_a_phase2_model(args.bert_feature_dim, args.ode_dim, args.num_classes, args.phase2_integration_time)

    saved_temp = torch.load(args.phase2_model_path)
    statedic_temp = saved_temp[list(saved_temp.keys())[0]] # ['model']
    phase2_model.load_state_dict(statedic_temp)
    phase2_model = phase2_model.to(device)

    if sanity_check: 
        print('Sanity Check on loaded model phase 1 ... ')
        trainloader, testloader = get_feature_dataloader(args, args.phase2_batch_size)
        criterion = get_loss(args.phase2_loss)
        tr_res = test_ce_one_epoch(-1, phase2_model, trainloader, device, criterion, 110, False, None, None)
        te_res = test_ce_one_epoch(-1, phase2_model, testloader, device, criterion, 110, False, None, None)
        print('Train Acc, Loss', tr_res['acc'], tr_res['loss'])
        print('Test Acc, Loss', te_res['acc'], te_res['loss'])

    return phase2_model

def train_phase3(phase2_model, args, device): 
    
    # parser.add_argument("--phase3_metric", nargs="+", default=["ACC"])
    
    assert args.ignore_dropout, 'if you dont ignore dropout, you need to do a drop out at the very begining of every model at the input data essentially' 
    assert args.phase3_optim == 'ADAM', 'Only Adam for now'
    
    save_path = os.path.join(args.output_dir, args.phase3_save_path)
    os.makedirs(save_path, exist_ok=True)

    trainloader, testloader = get_feature_dataloader(args, args.phase3_batch_size)


    ODE_layer = ODEBlock(odefunc=phase2_model.ode_block.odefunc, t = args.phase3_integration_time)
    
    phase3_model = Phase3Model(
        bridge_layer=phase2_model.bridge_layer, 
        ode_block=ODE_layer, 
        fc = MLP_OUT_LINEAR(args.ode_dim, args.num_classes) if not args.phase3_use_fc_from_phase2 else phase2_model.fc
    )
    phase3_model.set_all_req_grads(True)
    freeze_layers = []
    if args.phase3_freeze_backbone: 
        freeze_layers.append('bridge_layer')
    phase3_model.freeze_layer_given_name(freeze_layers)
    phase3_model = phase3_model.to(device)
    

    optimizer = torch.optim.Adam([{'params': phase3_model.ode_block.odefunc.parameters(), 'lr': args.phase3_lr_ode_block, 'eps':args.phase3_eps_ode_block,},
                                {'params': phase3_model.fc.parameters(), 'lr': args.phase3_lr_fc, 'eps':args.phase3_eps_fc_block,}], amsgrad=args.phase3_amsgrad)
    criterion = get_loss(args.phase3_loss)

    best_acc = 0 
    # args.phase3_metric
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    
    for epoch in trange(0, args.phase3_epochs):
        tr_results = train_ce_one_epoch(
            epoch=epoch, 
            model=phase3_model, 
            loader=trainloader, 
            device=device, 
            optimizer=optimizer,
            criterion=criterion
        )
        te_results = test_ce_one_epoch(
            epoch=epoch, 
            model=phase3_model, 
            loader=testloader,
            device=device,
            criterion=criterion,
            best_acc=best_acc,
            do_save=True,
            save_folder= save_path, # ?
            save_name= 'phase3' # ? 
        )
        best_acc = te_results['best_acc']

        print('tr_acc, tr_loss', tr_results['acc'], tr_results['loss'])
        print('te_acc, te_loss', te_results['acc'], te_results['loss'])
        
        train_loss_history.append(tr_results['loss'])
        train_acc_history.append(tr_results['acc'])
        
        test_loss_history.append(te_results['loss'])
        test_acc_history.append(te_results['acc'])

        wandb.log({
            'phase3/step': epoch, 
            'phase3/train_ce': tr_results['loss'], 'phase3/train_acc': tr_results['acc'], 
            'phase3/test_ce': te_results['loss'], 'phase3/test_acc': te_results['acc'], 
        })

    return phase3_model

def load_phase3(args, device, sanity_check = True): 
    saved_temp = torch.load(args.phase3_model_path)
    statedic_temp = saved_temp[list(saved_temp.keys())[0]] # ['model']

    phase3_model = get_a_phase3_model(args.bert_feature_dim, args.ode_dim, args.num_classes, args.phase3_integration_time)
    phase3_model.load_state_dict(statedic_temp)
    phase3_model = phase3_model.to(device)

    if sanity_check: 
        print('Sanity Check on loaded model phase 3 ... ')
        trainloader, testloader = get_feature_dataloader(args, args.phase1_batch_size)
        criterion = get_loss(args.phase1_loss)
        tr_res = test_ce_one_epoch(-1, phase3_model, trainloader, device, criterion, 110, False, None, None)
        te_res = test_ce_one_epoch(-1, phase3_model, testloader, device, criterion, 110, False, None, None)
        print('Train Acc, Loss', tr_res['acc'], tr_res['loss'])
        print('Test Acc, Loss', te_res['acc'], te_res['loss'])

    return phase3_model