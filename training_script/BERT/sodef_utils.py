import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import geotorch
from torchdiffeq import odeint_adjoint as odeint
import torch 
import math 

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

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

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

class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)

class ODEfunc_mlp(nn.Module): 

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 64)
        self.act1 = torch.sin 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out

class MLP_OUT_BALL(nn.Module):
    def __init__(self, dim1, dim2):
        super(MLP_OUT_BALL, self,).__init__()
        self.fc0 = nn.Linear(dim1, dim2, bias=False)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  
        

class Phase3Model(nn.Module): 
    def __init__(self, bridge_768_64, ode_block, fc): 
        super(Phase3Model, self).__init__()
        self.bridge_768_64 = bridge_768_64
        self.ode_block = ode_block
        self.fc = fc

    def set_all_req_grads(self, value=True):
        for name, param in self.named_parameters():
            param.requires_grad = True
            
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
        before_ode_feats = self.bridge_768_64(x)
        after_ode_feats = self.ode_block(before_ode_feats)
        logits = self.fc(after_ode_feats)
        return logits