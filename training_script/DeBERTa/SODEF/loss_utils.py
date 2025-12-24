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
    idx = np.random.choice(z.shape[0], size=numm, replace=False)
    print(z.shape[0], size=numm)
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

