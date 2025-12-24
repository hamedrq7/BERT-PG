import torch 
import numpy as np 
from torch.func import jacrev, vmap, jacfwd

def df_dz_regularizer(f, z, numm, odefunc, time_df, exponent, trans, exponent_off, transoffdig, device):
    # print("+++++++++++")
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1],-1)
        if batchijacobian.shape[0]!=batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")
            
        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent*(tempdiag+trans))
        offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
        off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
        regu_offdiag += off_diagtemp

    return regu_diag/numm, regu_offdiag/numm

def batched_df_dz_regularizer(f, z, numm, odefunc, time_df, exponent, trans, exponent_off, transoffdig, device):
    # Sample indices
    idx = np.random.choice(z.shape[0], size=min(numm,z.shape[0]), replace=False)
    # idx: [numm]

    # Compute batched Jacobian
    jacobian = vmap(jacrev(lambda x: odefunc(torch.tensor(time_df).to(device), x)))(z[idx])
    # batchijacobian: [numm, d, d]

    d = jacobian.shape[-1]

    # -------------------------
    # Diagonal regularizer
    # -------------------------
    diag = torch.diagonal(jacobian, dim1=1, dim2=2)
    # diag: [numm, d]

    regu_diag = torch.exp(exponent * (diag + trans))
    # regu_diag: [numm, d]
    regu_diag = regu_diag.sum(dim=0)
    # regu_diag: [d]

    # -------------------------
    # Off-diagonal regularizer
    # -------------------------
    eye = torch.eye(d, device=device)
    # eye: [d, d]

    offdiag_mask = ((-1 * eye + 0.5) * 2)
    # offdiag_mask: [d, d]
    # diagonal -> -1, off-diagonal -> 1

    offdiag = torch.abs(jacobian) * offdiag_mask
    # offdiag: [numm, d, d]

    offdiat = offdiag.sum(dim=1)
    # offdiat: [numm, d]

    regu_offdiag = torch.exp(exponent_off * (offdiat + transoffdig))
    # regu_offdiag: [numm, d]

    regu_offdiag = regu_offdiag.sum(dim=0)    
    # regu_offdiag: [d]

    # -------------------------
    # Normalize (same as original loop)
    # -------------------------
    return regu_diag / numm, regu_offdiag / numm



def f_regularizer(f, z, odefunc, time_df, device, exponent_f):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    # print('tempf: ', tempf.mean().item())
    
    return regu_f


from torch.autograd.function import Function
import torch.nn as nn 

class CenterLossNormal(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True, init_value = None):
        super(CenterLossNormal, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        if init_value is not None: 
            with torch.no_grad():
                self.centers.copy_(init_value)
        
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None
