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
import matplotlib.pyplot as plt 

@torch.no_grad()
def _choose_indices(N, num_points, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    return perm[: min(num_points, N)]


def jacobian_single(odefunc: nn.Module, z1x64: torch.Tensor, t: float = 1.0) -> torch.Tensor:
    """
    z1x64: shape [1, 64], requires_grad will be set inside.
    returns: J shape [64, 64]
    """
    assert z1x64.ndim == 2 and z1x64.shape[0] == 1 and z1x64.shape[1] == 64
    z = z1x64.clone().detach().requires_grad_(True)

    # autograd.functional.jacobian returns shape [1,64,1,64] for [1,64]->[1,64]
    J = torch.autograd.functional.jacobian(
        lambda x: odefunc(torch.tensor(t, device=x.device, dtype=x.dtype), x),
        z,
        create_graph=False,
        strict=True,
    )
    J = J.view(64, 64)
    return J

def spectrum_for_points(odefunc: nn.Module, feats: torch.Tensor, idx: torch.Tensor, device="cuda"):
    """
    feats: [N,64] float
    idx:  indices tensor
    returns:
        eigvals_all: complex tensor [M, 64]
        max_real:    float tensor [M]
    """
    odefunc = odefunc.to(device).eval()
    feats = feats.to(device)

    eigvals_list = []
    max_real_list = []

    for i in idx.tolist():
        z = feats[i:i+1]  # [1,64]
        J = jacobian_single(odefunc, z, t=1.0)  # [64,64]
        eigvals = torch.linalg.eigvals(J)       # [64] complex
        eigvals_list.append(eigvals)
        max_real_list.append(torch.max(eigvals.real))

    eigvals_all = torch.stack(eigvals_list, dim=0)  # [M,64]
    max_real = torch.stack(max_real_list, dim=0)    # [M]
    return eigvals_all, max_real


def summarize(eigvals_all: torch.Tensor, max_real: torch.Tensor, name: str):
    # eigvals_all: [M,64] complex
    re = eigvals_all.real.reshape(-1)
    im = eigvals_all.imag.reshape(-1)

    frac_pos = (re > 0).float().mean().item()
    frac_pos_strict = (re > 1e-6).float().mean().item()

    stats = {
        "name": name,
        "num_points": int(eigvals_all.shape[0]),
        "num_eigs_total": int(re.numel()),
        "frac_Re_gt_0": frac_pos,
        "frac_Re_gt_1e-6": frac_pos_strict,
        "max_real_mean": float(max_real.mean().item()),
        "max_real_median": float(max_real.median().item()),
        "max_real_p95": float(torch.quantile(max_real, 0.95).item()),
        "max_real_max": float(max_real.max().item()),
        "real_mean": float(re.mean().item()),
        "real_min": float(re.min().item()),
        "real_max": float(re.max().item()),
        "imag_abs_mean": float(im.abs().mean().item()),
    }
    return stats


def plot_complex_scatter(eigs_sodef, out_path="expD_eigs.png", title="Jacobian eigenspectrum at z(0)"):
    # Flatten
    rs = eigs_sodef.real.flatten().cpu().numpy()
    is_ = eigs_sodef.imag.flatten().cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.scatter(rs, is_, s=6, alpha=0.25, label="SODEF")
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")


def eigval_analysis(model, loader, device, output_path: str, phase: str,   
                    num_points=100, seed=111, ): 
    model = model.to(device)
    feats_before, feats_after, labels = model.collect_feats(loader, device)
    feats_before = feats_before.to(device)

    odefunc_sodef = model.ode_block.odefunc

    idx = _choose_indices(feats_before.shape[0], num_points, seed=seed)
    
    eigs_sodef, maxr_sodef = spectrum_for_points(odefunc_sodef, feats_before, idx, device=device)

    stats_sodef = summarize(eigs_sodef, maxr_sodef, "SODEF")

    print(f"\n=== Exp D summary - {phase} ===")
    for stats in (stats_sodef, ):
        print(f"\n[{stats['name']}]")
        for k, v in stats.items():
            if k == "name":
                continue
            print(f"  {k}: {v}")

    plot_complex_scatter(eigs_sodef, out_path=f'{output_path}/eigval_{phase}.png')