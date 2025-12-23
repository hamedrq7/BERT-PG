import torch 
import numpy as np 

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

def f_regularizer(f, z, odefunc, time_df, device, exponent_f):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    # print('tempf: ', tempf.mean().item())
    
    return regu_f

