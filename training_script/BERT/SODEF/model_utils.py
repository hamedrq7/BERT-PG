import torch 
import torch.nn as nn 
import numpy as np 
import os
import argparse
import logging
import time
import numpy as np
import torch
import timeit
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
import math 
from torchdiffeq import odeint_adjoint as odeint
from PIL import Image


class Phase1Model(nn.Module): 
    def __init__(self, feature_dim, ode_dim, num_classes): 
        super(Phase1Model, self).__init__()
        self.orthogonal_bridge_layer = MLP_OUT_ORTH_X_X(feature_dim, ode_dim) 
        max_row_mat = get_max_row_dist_for_2_classes(ode_dim)
        check_max_row_dist_matrix(max_row_mat.cpu().numpy().T, 2)
        self.fc = MLP_OUT_BALL_given_mat(max_row_mat, dim=ode_dim, num_classes=num_classes)
    
    def set_all_req_grads(self, value=True):
        for name, param in self.named_parameters():
            param.requires_grad = value
            
    def forward(self, x): 
        x = self.orthogonal_bridge_layer(x)
        x = self.fc(x)
        return x

def get_a_phase1_model(feature_dim, ode_dim, num_classes):
    return Phase1Model(feature_dim, ode_dim, num_classes) 

def check_max_row_dist_matrix(V, num_classes = 10):
    assert V.shape[1] == num_classes, 'V should be of shape [D, num_classes] '

    # --------------------------------------
    # 1. Check unit norms
    # --------------------------------------
    norms = np.linalg.norm(V, axis=0)
    print("Column norms:")
    print(norms)

    # --------------------------------------
    # 2. Compute cosine similarity matrix
    # --------------------------------------
    cos_sim = V.T @ V   # (10×10 matrix)
    n_classes = V.shape[1]
    desired_ip = 1.0 / (1.0 - n_classes)
    print(f"\nCosine similarity matrix (desired off diag={desired_ip}):")
    print(cos_sim)

    # --------------------------------------
    # 3. Extract off-diagonal values
    # --------------------------------------
    off_diag = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]

    # Print statistics
    print("\nOff-diagonal cosine similarity statistics:")
    print(f"mean: {off_diag.mean():.6f}")
    print(f"std : {off_diag.std():.6f}")
    print(f"min : {off_diag.min():.6f}")
    print(f"max : {off_diag.max():.6f}")

    # Optional: check if all similarities are equal (within tolerance)
    if np.allclose(off_diag, off_diag[0], atol=1e-4):
        print("\n✔ All off-diagonal similarities are equal.")
    else:
        print("\n✘ Off-diagonal similarities are NOT equal.")


def get_max_row_dist_for_2_classes(dim = 64, device="cpu"):
    # random unit vector in R^64
    u = torch.randn(dim, device=device)
    u = u / u.norm()

    v1 = u
    v2 = -u

    # shape = (feature_dim, 2)
    W = torch.stack([v1, v2], dim=1)
    return W.T

class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)

class MLP_OUT_ORTH_X_X(nn.Module):
    def __init__(self, dim1, dim2):
        super(MLP_OUT_ORTH_X_X, self).__init__()
        self.fc0 = ORTHFC(dim1, dim2, False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
    
class MLP_OUT_BALL_given_mat(nn.Module):
    def __init__(self, mat, dim, num_classes):
        super(MLP_OUT_BALL_given_mat, self,).__init__()
        assert mat.shape[1] == dim and mat.shape[0] == num_classes, 'This should hold: mat.shape[0] == dim and mat.shape[1] == num_classes'
        self.fc0 = nn.Linear(dim, num_classes, bias=False)
        matrix_temp = torch.tensor(mat)

        self.fc0.weight.data = matrix_temp
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  

################################### 

class Phase2Model(nn.Module): 
    def __init__(self, bridge_layer, ode_block, fc): 
        super(Phase2Model, self).__init__()
        self.bridge_layer = bridge_layer
        self.ode_block = ode_block
        self.fc = fc

    def set_all_req_grads(self, value=True):
        for name, param in self.named_parameters():
            param.requires_grad = value
            
    def freeze_layer_given_name(self, layer_names): 
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for target in layer_names:
            if target in self._modules:   # only top-level modules
                module = self._modules[target]
                for param in module.parameters():
                    param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad == True: 
                print(f"[TRAINABLE] {name}")
            else:
                print(f"[FROZEN] {name}")
    
    def forward(self, x): 
        before_ode_feats = self.bridge_layer(x)
        after_ode_feats = self.ode_block(before_ode_feats)
        logits = self.fc(after_ode_feats)
        return before_ode_feats, after_ode_feats, logits

def get_a_phase2_model(feature_dim, ode_dim, num_classes, t): 
    dummy_phase1 = get_a_phase1_model(feature_dim, ode_dim, num_classes)
    odefunc = ODEfunc_mlp(ode_dim)
    phase2_model = Phase2Model(
        dummy_phase1.orthogonal_bridge_layer, 
        ODEBlocktemp(odefunc, t), 
        dummy_phase1.fc
    )
    return phase2_model

class SingleOutputWrapper(nn.Module): 
    def __init__(self, model): 
        super(SingleOutputWrapper, self).__init__()
        self.model = model 

    def forward(self, x): 
        _, _, out = self.model(x)
        return out 



class ODEBlocktemp(nn.Module):  ####  note here we do not integrate to save time

    def __init__(self, odefunc, integration_time: float):
        super(ODEBlocktemp, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, integration_time]).float()

    def forward(self, x):
        out = self.odefunc(0, x)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


class ODEfunc_mlp(nn.Module): 

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(dim, dim)
        self.act1 = torch.sin 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out

class MLP_OUT_LINEAR(nn.Module):
    def __init__(self, dim1=64, dim2=10):
        super(MLP_OUT_LINEAR, self, ).__init__()
        self.fc0 = nn.Linear(dim1, dim2, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
    

################### 
class Phase3Model(nn.Module): 
    def __init__(self, bridge_layer, ode_block, fc): 
        super(Phase3Model, self).__init__()
        self.bridge_layer = bridge_layer
        self.ode_block = ode_block
        self.fc = fc

    def set_all_req_grads(self, value=True):
        for name, param in self.named_parameters():
            param.requires_grad = value
            
    def freeze_layer_given_name(self, layer_names): 
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        for target in layer_names:
            if target in self._modules:   # only top-level modules
                module = self._modules[target]
                for param in module.parameters():
                    param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad == True: 
                print(f"[TRAINABLE] {name}")
            else:
                print(f"[FROZEN] {name}")
    
    def forward(self, x): 
        before_ode_feats = self.bridge_layer(x)
        after_ode_feats = self.ode_block(before_ode_feats)
        logits = self.fc(after_ode_feats)
        return logits

def get_a_phase3_model(feature_dim, ode_dim, num_classes, t):
    dummy = get_a_phase2_model(feature_dim, ode_dim, num_classes, t)
    ODE_layer = ODEBlock(odefunc=dummy.ode_block.odefunc, t = t)

    phase3_model = Phase3Model(
        bridge_layer=dummy.bridge_layer, 
        ode_block=ODE_layer, 
        fc = dummy.fc
    )
    return phase3_model

class ODEBlock(nn.Module):

    def __init__(self, odefunc, t):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, t]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value   