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
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
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
            
    #### need tro make dir the output path 
    # plot_complex_scatter(eigs_sodef, out_path=f'{output_path}/eigval_{phase}.png')



# -----------------------------
# Helpers
# -----------------------------
def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def _hist_three(values: torch.Tensor, correct_mask: torch.Tensor, title: str, xlabel: str, out_path: str, bins: int = 60):
    v = _to_cpu(values).flatten()
    c = _to_cpu(correct_mask).flatten().bool()
    v_all = v
    v_cor = v[c]
    v_inc = v[~c]

    plt.figure(figsize=(7, 5))
    plt.hist(v_all.numpy(), bins=bins, alpha=0.45, label=f"all (n={len(v_all)})")
    if len(v_cor) > 0:
        plt.hist(v_cor.numpy(), bins=bins, alpha=0.45, label=f"correct (n={len(v_cor)})")
    if len(v_inc) > 0:
        plt.hist(v_inc.numpy(), bins=bins, alpha=0.45, label=f"incorrect (n={len(v_inc)})")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    _savefig(out_path)

def _bar(values: torch.Tensor, title: str, xlabel: str, ylabel: str, out_path: str):
    v = _to_cpu(values).flatten()
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(v)), v.numpy())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _savefig(out_path)

def _imshow(mat: torch.Tensor, title: str, out_path: str, vmin: Optional[float] = None, vmax: Optional[float] = None):
    m = _to_cpu(mat)
    plt.figure(figsize=(6, 5))
    plt.imshow(m.numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("class")
    plt.ylabel("class")
    _savefig(out_path)


# -----------------------------
# Core analysis
# -----------------------------
@dataclass
class SodefAnalysisInputs:
    before_feats: torch.Tensor              # [N,d]
    after_feats: torch.Tensor               # [N,d]
    labels: torch.Tensor                    # [N]
    pred_before: torch.Tensor               # [N]
    pred_after: torch.Tensor                # [N]
    loss_before: torch.Tensor               # [N] per-sample
    loss_after: torch.Tensor                # [N] per-sample
    final_layer: nn.Linear                  # Linear(d,C)
    raw_logits: torch.Tensor 
    denoised_logits: torch.Tensor 

def analyze_final_layer(final_layer: nn.Linear, out_dir: str) -> Dict:
    W = final_layer.weight.detach()  # [C,d]
    b = final_layer.bias.detach() if final_layer.bias is not None else None
    C, d = W.shape

    # Weight norms & bias
    w_norm = W.norm(dim=1)
    _bar(w_norm, "Final-layer weight norms per class", "class", "||w_c||", os.path.join(out_dir, "final_w_norms.png"))
    stats = {
        "num_classes": C,
        "feat_dim": d,
        "w_norm_mean": float(w_norm.mean().item()),
        "w_norm_min": float(w_norm.min().item()),
        "w_norm_max": float(w_norm.max().item()),
    }

    if b is not None:
        _bar(b, "Final-layer bias per class", "class", "bias", os.path.join(out_dir, "final_bias.png"))
        stats.update({
            "bias_mean": float(b.mean().item()),
            "bias_min": float(b.min().item()),
            "bias_max": float(b.max().item()),
        })

    # Orthogonality / separation of class weights: cosine matrix
    Wn = _normalize(W)
    cosW = Wn @ Wn.t()  # [C,C]
    _imshow(cosW, "Cosine similarity between class weight vectors (W)", os.path.join(out_dir, "final_w_cos_matrix.png"),
            vmin=-1.0, vmax=1.0)

    # Off-diagonal summary
    off = cosW.clone()
    off.fill_diagonal_(0.0)
    off_vals = off.flatten()
    stats.update({
        "w_cos_offdiag_mean": float(off_vals.mean().item()),
        "w_cos_offdiag_abs_mean": float(off_vals.abs().mean().item()),
        "w_cos_offdiag_max": float(off_vals.max().item()),
        "w_cos_offdiag_min": float(off_vals.min().item()),
        "w_cos_offdiag_abs_max": float(off_vals.abs().max().item()),
    })

    # Histogram of off-diagonal cosines
    plt.figure(figsize=(7, 5))
    plt.hist(_to_cpu(off_vals).numpy(), bins=60, alpha=0.85)
    plt.title("Histogram of off-diagonal cosines between class weights")
    plt.xlabel("cos(w_i, w_j), i≠j")
    plt.ylabel("count")
    _savefig(os.path.join(out_dir, "final_w_offdiag_cos_hist.png"))

    return stats

def per_stage_feature_analysis(
    stage_name: str,
    feats: torch.Tensor,          # [N,d]
    labels: torch.Tensor,         # [N]
    preds: torch.Tensor,          # [N]
    final_layer: nn.Linear,
    out_dir: str
) -> Dict:
    _ensure_dir(out_dir)
    device = feats.device
    labels = labels.to(device)
    preds = preds.to(device)

    correct = (preds == labels)

    # 1) Norm hist (all/correct/incorrect)
    feat_norm = feats.norm(dim=1)
    _hist_three(
        feat_norm, correct,
        title=f"{stage_name}: ||feature|| distribution",
        xlabel="L2 norm",
        out_path=os.path.join(out_dir, f"{stage_name}_feat_norm_hist.png"),
    )

    # 2) Norm of mean feature per class
    C = final_layer.out_features
    d = feats.shape[1]
    mean_norms = torch.zeros(C, device=device)
    counts = torch.zeros(C, device=device)
    for c in range(C):
        m = feats[labels == c]
        counts[c] = m.shape[0]
        if m.shape[0] > 0:
            mean_norms[c] = m.mean(dim=0).norm()
        else:
            mean_norms[c] = float("nan")

    _bar(mean_norms, f"{stage_name}: ||mean feature|| per class", "class", "||E[f|y=c]||",
         os.path.join(out_dir, f"{stage_name}_mean_feat_norm_bar.png"))

    # 3) Intra-class angular variance (cosine to class mean direction)
    #    Use mean of normalized features as class direction.
    feats_n = _normalize(feats)
    class_dir = torch.zeros(C, d, device=device)
    ang_var = torch.zeros(C, device=device)
    ang_mean = torch.zeros(C, device=device)

    for c in range(C):
        idx = (labels == c)
        if idx.sum() < 2:
            ang_var[c] = float("nan")
            ang_mean[c] = float("nan")
            continue
        fc = feats_n[idx]  # normalized
        mu = _normalize(fc.mean(dim=0, keepdim=True)).squeeze(0)  # unit
        class_dir[c] = mu
        cos_to_mu = (fc @ mu)  # [Nc]
        ang_mean[c] = cos_to_mu.mean()
        ang_var[c] = cos_to_mu.var(unbiased=False)

    _bar(ang_var, f"{stage_name}: intra-class cosine variance to class mean direction",
         "class", "Var(cos(f̂, μ̂_c))",
         os.path.join(out_dir, f"{stage_name}_intra_class_cos_var_bar.png"))

    # Extra A) Feature alignment to *true* class weight
    Wn = _normalize(final_layer.weight.detach().to(device))  # [C,d]
    true_w = Wn[labels]                                     # [N,d]
    align_true = (feats_n * true_w).sum(dim=1)              # [N]
    _hist_three(
        align_true, correct,
        title=f"{stage_name}: cos(feature, w_true) distribution",
        xlabel="cos(f̂, ŵ_y)",
        out_path=os.path.join(out_dir, f"{stage_name}_align_true_hist.png"),
    )

    # Extra B) Margins / confidence from the linear layer
    logits = final_layer(feats)                              # [N,C]
    y = labels
    true_log = logits[torch.arange(logits.shape[0], device=device), y]
    tmp = logits.clone()
    tmp[torch.arange(logits.shape[0], device=device), y] = -1e9
    max_other = tmp.max(dim=1).values
    margin = true_log - max_other

    probs = torch.softmax(logits, dim=1)
    conf = probs.max(dim=1).values

    _hist_three(margin, correct, f"{stage_name}: margin distribution", "margin (logit_y - max_other)",
                os.path.join(out_dir, f"{stage_name}_margin_hist.png"))
    _hist_three(conf, correct, f"{stage_name}: confidence distribution", "max softmax prob",
                os.path.join(out_dir, f"{stage_name}_confidence_hist.png"))

    stats = {
        "acc": float(correct.float().mean().item()),
        "feat_norm_mean": float(feat_norm.mean().item()),
        "feat_norm_median": float(feat_norm.median().item()),
        "align_true_mean": float(align_true.mean().item()),
        "margin_mean": float(margin.mean().item()),
        "conf_mean": float(conf.mean().item()),
        "class_counts": _to_cpu(counts).tolist(),
        "intra_cos_var_mean": float(torch.nanmean(ang_var).item()) if torch.isnan(ang_var).any() else float(ang_var.mean().item()),
    }
    return stats

def analyze_feature_transition(
    before_feats: torch.Tensor,
    after_feats: torch.Tensor,
    labels: torch.Tensor,
    pred_after: torch.Tensor,
    loss_before: torch.Tensor,
    loss_after: torch.Tensor,
    out_dir: str,
    prefix: str = "transition"
) -> Dict:
    _ensure_dir(out_dir)
    device = before_feats.device
    labels = labels.to(device)
    pred_after = pred_after.to(device)
    correct = (pred_after == labels)  # choose correctness definition here (after-ODE)

    # (2) cosine_similarity(f_b, f_a)
    fb = _normalize(before_feats)
    fa = _normalize(after_feats)
    cos_ba = (fb * fa).sum(dim=1)

    _hist_three(cos_ba, correct,
                f"{prefix}: cos(before, after) distribution",
                "cos(f̂_before, f̂_after)",
                os.path.join(out_dir, f"{prefix}_cos_before_after_hist.png"))

    # (3) norm(f_a - f_b)
    delta = after_feats - before_feats
    delta_norm = delta.norm(dim=1)
    _hist_three(delta_norm, correct,
                f"{prefix}: ||after - before|| distribution",
                "||f_after - f_before||",
                os.path.join(out_dir, f"{prefix}_delta_norm_hist.png"))

    # (4) loss difference hist
    dloss = (loss_after - loss_before).detach()
    _hist_three(dloss, correct,
                f"{prefix}: Δloss = loss_after - loss_before",
                "Δloss",
                os.path.join(out_dir, f"{prefix}_dloss_hist.png"))

    # Also: histogram of raw losses if you want
    _hist_three(loss_before, correct,
                f"{prefix}: loss_before distribution",
                "loss_before",
                os.path.join(out_dir, f"{prefix}_loss_before_hist.png"))
    _hist_three(loss_after, correct,
                f"{prefix}: loss_after distribution",
                "loss_after",
                os.path.join(out_dir, f"{prefix}_loss_after_hist.png"))

    stats = {
        "cos_before_after_mean": float(cos_ba.mean().item()),
        "cos_before_after_median": float(cos_ba.median().item()),
        "delta_norm_mean": float(delta_norm.mean().item()),
        "dloss_mean": float(dloss.mean().item()),
        "dloss_frac_negative": float((dloss < 0).float().mean().item()),
    }
    return stats

def run_full_sodef_analysis(inp: SodefAnalysisInputs, out_dir: str) -> Dict:
    _ensure_dir(out_dir)

    # Basic accuracy compare (your bullet)
    acc_before = float((inp.pred_before == inp.labels).float().mean().item())
    acc_after = float((inp.pred_after == inp.labels).float().mean().item())

    results = {
        "acc_before": acc_before,
        "acc_after": acc_after,
        "acc_delta": acc_after - acc_before,
    }

    # Final layer geometry
    results["final_layer"] = analyze_final_layer(inp.final_layer, os.path.join(out_dir, "final_layer"))

    # Feature space: before vs after
    results["before_feats"] = per_stage_feature_analysis(
        "before", inp.before_feats, inp.labels, inp.pred_before, inp.final_layer, os.path.join(out_dir, "before_feats")
    )
    results["after_feats"] = per_stage_feature_analysis(
        "after", inp.after_feats, inp.labels, inp.pred_after, inp.final_layer, os.path.join(out_dir, "after_feats")
    )

    # Transition before -> after
    results["transition"] = analyze_feature_transition(
        inp.before_feats, inp.after_feats, inp.labels, inp.pred_after, inp.loss_before, inp.loss_after,
        os.path.join(out_dir, "transition"), prefix="transition"
    )

    # Also: overall histogram of Δloss (your earlier bullet)
    dloss = (inp.loss_after - inp.loss_before).detach()
    plt.figure(figsize=(7, 5))
    plt.hist(_to_cpu(dloss).numpy(), bins=80, alpha=0.85)
    plt.title("Histogram: Δloss = loss_after - loss_before (all samples)")
    plt.xlabel("Δloss")
    plt.ylabel("count")
    _savefig(os.path.join(out_dir, "dloss_all_hist.png"))

    return results

def _denoising_analysis(model, phase, loader, device): 
    print('*'*20)
    print('_denoising_analysis', phase)
    print('*'*20)

    data: SodefAnalysisInputs = model.collect_feats(loader, device, return_outputs = True)

    results = run_full_sodef_analysis(data, out_dir=f"./sodef_analysis_{phase}")
    print(results["acc_before"], results["acc_after"], results["acc_delta"])
    print(results["transition"])

def denoising_analysis(phase2_model, phase3_model, train_loader, test_loader, device, adv_loader = None): 
    if phase2_model is not None: 
        _denoising_analysis(phase2_model, 'phase2_train', train_loader, device)
        _denoising_analysis(phase2_model, 'phase2_test', test_loader, device)
        if adv_loader is not None: 
            _denoising_analysis(phase2_model, 'phase2_adv_glue', adv_loader, device)
    
    _denoising_analysis(phase3_model, 'phase3_train', train_loader, device)
    _denoising_analysis(phase3_model, 'phase3_test', test_loader, device)
    if adv_loader is not None: 
        _denoising_analysis(phase3_model, 'phase3_adv_glue', adv_loader, device)
    
