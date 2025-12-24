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
from loss_utils import df_dz_regularizer, f_regularizer, batched_df_dz_regularizer
from model_utils import SingleOutputWrapper
from model_utils import ODEBlock, Phase3Model, get_a_phase1_model, get_a_phase2_model, get_a_phase3_model
import wandb
import matplotlib.pyplot as plt 
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from model_utils import SodefAnalysisInputs

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


def online_eigval_analysis(model, feats, device, num_points=100, seed=111, ): 
    feats = feats.to(device)

    odefunc_sodef = model.ode_block.odefunc

    idx = _choose_indices(feats.shape[0], num_points, seed=seed)
    
    eigs_sodef, maxr_sodef = spectrum_for_points(odefunc_sodef, feats, idx, device=device)

    stats_sodef = summarize(eigs_sodef, maxr_sodef, "SODEF")

    return stats_sodef

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
    plt.hist(
        [v_all.numpy(), v_cor.numpy(), v_inc.numpy()],
        bins=bins,
        alpha=0.45,
        color=["tab:blue", "tab:orange", "tab:green"],
        label=[f"all (n={len(v_all)})", f"correct (n={len(v_cor)})", f"incorrect (n={len(v_inc)})"],
    )

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
    logits: torch.Tensor, 
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


def add_alignment_delta_hist_by_class(
    before_feats: torch.Tensor,    # [N,d]
    after_feats: torch.Tensor,     # [N,d]
    labels: torch.Tensor,          # [N]
    final_layer,                   # nn.Linear(d,C)
    out_dir: str,
    filename: str = "delta_cos_feat_to_true_w_by_class.png",
    bins: int = 60,
):
    """
    Computes Δ = cos(f_after, w_y) - cos(f_before, w_y) for each sample,
    where w_y is the final-layer weight of the sample's (true) class y.
    Plots a histogram with one distribution per class (assumes 2 classes).
    Returns the per-sample deltas [N].
    """
    os.makedirs(out_dir, exist_ok=True)
    device = before_feats.device

    y = labels.to(device).long()
    W = final_layer.weight.detach().to(device)  # [C,d]
    C, d = W.shape
    assert C == 2, f"This plotting helper expects 2 classes, got C={C}."

    # Normalize feats + weights
    eps = 1e-12
    bf = before_feats / (before_feats.norm(dim=1, keepdim=True) + eps)
    af = after_feats / (after_feats.norm(dim=1, keepdim=True) + eps)
    Wn = W / (W.norm(dim=1, keepdim=True) + eps)

    wy = Wn[y]  # [N,d] weight vector for each sample's class

    cos_before = (bf * wy).sum(dim=1)  # [N]
    cos_after  = (af * wy).sum(dim=1)  # [N]
    delta = cos_after - cos_before     # [N]

    # Plot: two histograms (class 0 and class 1)
    delta_cpu = delta.detach().cpu()
    y_cpu = y.detach().cpu()

    d0 = delta_cpu[y_cpu == 0].numpy()
    d1 = delta_cpu[y_cpu == 1].numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(d0, bins=bins, alpha=0.45, label=f"class 0 (n={len(d0)})")
    plt.hist(d1, bins=bins, alpha=0.45, label=f"class 1 (n={len(d1)})")
    plt.axvline(0.0, linewidth=1.0)
    plt.title("Δ alignment to true class weight: cos(f_after,w_y) − cos(f_before,w_y)")
    plt.xlabel("Δ cosine alignment")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()

    return delta, out_path

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
    _ensure_dir(os.path.join(out_dir, "final_layer"))
    results["final_layer"] = analyze_final_layer(inp.final_layer, os.path.join(out_dir, "final_layer"))

    # Feature space: before vs after
    _ensure_dir(os.path.join(out_dir, "before_feats"))
    results["before_feats"] = per_stage_feature_analysis(
        "before", inp.before_feats, inp.labels, inp.pred_before, inp.final_layer, inp.raw_logits, os.path.join(out_dir, "before_feats")
    )
    _ensure_dir(os.path.join(out_dir, "after_feats"))
    results["after_feats"] = per_stage_feature_analysis(
        "after", inp.after_feats, inp.labels, inp.pred_after, inp.final_layer, inp.denoised_logits, os.path.join(out_dir, "after_feats")
    )

    # Transition before -> after
    _ensure_dir(os.path.join(out_dir, "transition"))
    results["transition"] = analyze_feature_transition(
        inp.before_feats, inp.after_feats, inp.labels, inp.pred_after, inp.loss_before, inp.loss_after,
        os.path.join(out_dir, "transition"), prefix="transition"
    )
    results["delta_align_true_w"], _ = add_alignment_delta_hist_by_class(
        inp.before_feats, inp.after_feats, inp.labels, inp.final_layer,
        out_dir=os.path.join(out_dir, "transition")
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
    
import torch
import torch.nn.functional as F


def pgd_l2_untargeted_early_stop(
    x: torch.Tensor,          # [1,d]
    y0: torch.Tensor,         # [1] label you want to KEEP (original prediction)
    forward_fn,               # maps x -> logits [1,C]
    eps: float,
    steps: int = 40,
    step_size: float = 0.1,
    random_start: bool = True,
) -> Tuple[torch.Tensor, bool]:
    """
    Untargeted PGD in L2 ball with early stopping:
    - tries to find x_adv within ||x_adv - x||_2 <= eps that flips prediction away from y0
    - stops immediately when pred != y0
    Returns: (x_adv, success)
    """
    x0 = x.detach()
    device = x0.device

    # init
    if random_start and eps > 0:
        delta = torch.randn_like(x0)
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-12)
        delta = delta * (0.5 * eps)
        x_adv = (x0 + delta).detach()
    else:
        x_adv = x0.clone().detach()

    # quick check
    with torch.no_grad():
        if forward_fn(x_adv).argmax(dim=1).ne(y0).item():
            return x_adv, True

    x_adv.requires_grad_(True)

    for _ in range(steps):
        logits = forward_fn(x_adv)                 # [1,C]
        loss = F.cross_entropy(logits, y0)         # maximize this -> move away from class y0
        grad = torch.autograd.grad(loss, x_adv)[0]

        # L2-normalized gradient ascent step
        g = grad / (grad.norm(dim=1, keepdim=True) + 1e-12)
        x_adv = x_adv.detach() + step_size * g

        # project back to L2 ball
        delta = x_adv - x0
        norm = delta.norm(dim=1, keepdim=True) + 1e-12
        factor = torch.clamp(eps / norm, max=1.0)
        x_adv = (x0 + delta * factor).detach()

        # early stop check
        with torch.no_grad():
            if forward_fn(x_adv).argmax(dim=1).ne(y0).item():
                return x_adv, True

        x_adv.requires_grad_(True)

    return x_adv.detach(), False

@torch.no_grad()
def flips_prediction(x_adv: torch.Tensor, y_orig: torch.Tensor, forward_fn) -> bool:
    pred = forward_fn(x_adv).argmax(dim=1)
    return bool((pred != y_orig).item())

from tqdm import trange 

def empirical_radius_full_model_early_stop(
    feats: torch.Tensor,      # [N,d] (ODE input features)
    forward_fn,               # x->[N,C] full pipeline logits (ODE+FC)
    eps_max: float = 5.0,
    bin_steps: int = 12,
    pgd_steps: int = 40,
    pgd_step_size: float = 0.1,
) -> torch.Tensor:
    """
    Binary search over eps with PGD early stopping at each eps.
    Returns approximate minimal L2 radius per sample to flip prediction.
    """
    device = feats.device
    N = feats.shape[0]
    radii = torch.empty(N, device=device)

    # original prediction labels (the label you want to keep)
    with torch.no_grad():
        y0_all = forward_fn(feats).argmax(dim=1)

    for i in trange(N):
        x = feats[i:i+1]
        y0 = y0_all[i:i+1]

        lo, hi = 0.0, eps_max

        # Optional: if it doesn't flip even at eps_max, you'll end at eps_max.
        for _ in range(bin_steps):
            mid = 0.5 * (lo + hi)
            _, success = pgd_l2_untargeted_early_stop(
                x, y0, forward_fn,
                eps=mid,
                steps=pgd_steps,
                step_size=pgd_step_size,
                random_start=False,
            )
            if success:
                hi = mid
            else:
                lo = mid

        radii[i] = hi

    return radii

import matplotlib.pyplot as plt

def plot_radius_hist(radii, correct_mask, out_png, bins=60, title="Robustness radius"):
    r = radii.detach().cpu()
    c = correct_mask.detach().cpu().bool()

    plt.figure(figsize=(7,5))
    plt.hist(r.numpy(), bins=bins, alpha=0.45, label=f"all (n={len(r)})")
    plt.hist(r[c].numpy(), bins=bins, alpha=0.45, label=f"correct (n={int(c.sum())})")
    plt.hist(r[~c].numpy(), bins=bins, alpha=0.45, label=f"incorrect (n={int((~c).sum())})")
    plt.title(title)
    plt.xlabel("minimal L2 perturbation to flip prediction")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def raddddddi(model, phase, loader, device): 
    data: SodefAnalysisInputs = model.collect_feats(loader, device, return_outputs = True)

    n = 150
    feats = data.before_feats[0:n]
    labels = data.labels[0:n]

    def forward_fn(x):
        x = x.to(device)
        z = model.ode_block(x)          # [N,64]
        logits = model.fc(z)   # [N,C]
        return logits

    EPS = 10.0
    STEPS = 5
    STEP_SIZE = EPS / STEPS * 2

    radii = empirical_radius_full_model_early_stop(
        feats=feats,
        forward_fn=forward_fn,
        eps_max=EPS,        # adjust if needed
        bin_steps=20,
        pgd_steps=STEPS,
        pgd_step_size=STEP_SIZE,  # can tune
    )

    with torch.no_grad():
        pred = forward_fn(feats).argmax(dim=1)
    correct_mask = (pred.cpu() == labels)

    plot_radius_hist(
        radii=radii,
        correct_mask=correct_mask,
        out_png=f"{phase}-robust_radius_hist.png",
        title="Min L2 perturbation (on ODE-input feats) to flip full-model prediction",
    )

import torch

def sample_l2_sphere_noise(shape, eps: float, device=None, dtype=None):
    # Gaussian -> normalize -> scale to eps
    delta = torch.randn(shape, device=device, dtype=dtype)
    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-12)
    return delta * eps

def sample_l2_ball_noise(shape, eps: float, device=None, dtype=None):
    # Sample on sphere then scale radius ~ U(0,1)^(1/d)
    d = shape[-1]
    delta = sample_l2_sphere_noise(shape, eps=1.0, device=device, dtype=dtype)
    u = torch.rand(shape[:-1] + (1,), device=device, dtype=dtype)
    r = (u ** (1.0 / d)) * eps
    return delta * r

@torch.no_grad()
def random_robustness_prob(
    feats: torch.Tensor,        # [N,d] ODE input feats
    forward_fn,                 # x->[N,C] logits for full pipeline
    eps: float,
    K: int = 64,                # #random draws per sample
    chunk_K: int = 16,          # do K in chunks to limit memory
    use_ball: bool = False,     # sphere vs ball
):
    device = feats.device
    N, d = feats.shape

    # baseline prediction to preserve
    y0 = forward_fn(feats).argmax(dim=1)   # [N]

    keep_counts = torch.zeros(N, device=device, dtype=torch.float32)

    sampler = sample_l2_ball_noise if use_ball else sample_l2_sphere_noise

    done = 0
    while done < K:
        k = min(chunk_K, K - done)  # this chunk count

        # Create [N,k,d] noise and evaluate in one big batch [N*k,d]
        noise = sampler((N, k, d), eps=eps, device=device, dtype=feats.dtype)
        x_pert = feats[:, None, :] + noise                  # [N,k,d]
        x_pert = x_pert.reshape(N * k, d)                   # [N*k,d]

        pred = forward_fn(x_pert).argmax(dim=1).view(N, k)  # [N,k]
        keep = pred.eq(y0[:, None])                         # [N,k]
        keep_counts += keep.float().sum(dim=1)

        done += k

    p_keep = keep_counts / float(K)  # [N]
    return p_keep

import torch

@torch.no_grad()
def random_robustness_prob_dual_relative(
    feats: torch.Tensor,      # [N,d] ODE-input features
    forward_fn,               # returns (raw_logits, denoised_logits)
    alpha: float,             # relative noise: ||δ_i|| = alpha * ||x_i||
    labels, 
    K: int = 64,
    chunk_K: int = 16,
):
    """
    Returns:
      p_keep_raw:      [N]  fraction of random perturbations that keep RAW prediction
      p_keep_denoised: [N]  fraction of random perturbations that keep DENOISED prediction
    """
    device = feats.device
    N, d = feats.shape

    # Baseline predictions to preserve (separately for raw and denoised)
    raw0, den0 = forward_fn(feats)
    # y0_raw = raw0.argmax(dim=1)  # [N]
    # y0_den = den0.argmax(dim=1)  # [N]
    y0_raw = labels
    y0_den = labels 

    # Per-sample epsilon vector: eps_i = alpha * ||x_i||
    eps = alpha * feats.norm(dim=1)                 # [N]
    eps = eps.clamp(min=0.0)
    eps_vec = eps.view(N, 1, 1)                     # [N,1,1] for broadcasting

    keep_raw = torch.zeros(N, device=device, dtype=torch.float32)
    keep_den = torch.zeros(N, device=device, dtype=torch.float32)

    done = 0
    while done < K:
        k = min(chunk_K, K - done)

        # noise: [N,k,d] with unit norm then scaled per-sample
        delta = torch.randn((N, k, d), device=device, dtype=feats.dtype)
        delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-12)
        delta = delta * eps_vec  # broadcast -> [N,k,d], each sample has its own eps_i

        x_pert = feats[:, None, :] + delta          # [N,k,d]
        x_pert = x_pert.reshape(N * k, d)           # [N*k,d]

        raw_logits, den_logits = forward_fn(x_pert)
        pred_raw = raw_logits.argmax(dim=1).view(N, k)
        pred_den = den_logits.argmax(dim=1).view(N, k)

        keep_raw += pred_raw.eq(y0_raw[:, None]).float().sum(dim=1)
        keep_den += pred_den.eq(y0_den[:, None]).float().sum(dim=1)

        done += k

    p_keep_raw = keep_raw / float(K)
    p_keep_den = keep_den / float(K)
    return p_keep_raw, p_keep_den

import matplotlib.pyplot as plt
import os

def plot_pkeep_hist(p_keep: torch.Tensor, correct_mask: torch.Tensor, out_png: str, title: str, bins: int = 50):
    p = p_keep.detach().cpu()
    c = correct_mask.detach().cpu().bool()

    plt.figure(figsize=(7,5))
    plt.hist(p.numpy(), bins=bins, alpha=0.45, label=f"all (n={len(p)})")
    plt.hist(p[c].numpy(), bins=bins, alpha=0.45, label=f"correct (n={int(c.sum())})")
    plt.hist(p[~c].numpy(), bins=bins, alpha=0.45, label=f"incorrect (n={int((~c).sum())})")
    plt.title(title)
    plt.xlabel("P[pred stays same under random perturbation]")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_two_hist(values_a: torch.Tensor, values_b: torch.Tensor, label_a: str, label_b: str,
                 title: str, xlabel: str, out_png: str, bins: int = 50):
    a = values_a.detach().cpu().numpy()
    b = values_b.detach().cpu().numpy()

    plt.figure(figsize=(7,5))
    
    plt.hist(
        [a, b],
        bins=bins,
        alpha=0.45,
        color=["tab:blue", "tab:orange"],
        label=[f"{label_a} (n={len(a)})", f"{label_b} (n={len(b)})"],
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_delta_hist(values_den: torch.Tensor, values_raw: torch.Tensor, title: str, out_png: str, bins: int = 60):
    d = (values_den - values_raw).detach().cpu().numpy()
    plt.figure(figsize=(7,5))
    plt.hist(d, bins=bins, alpha=0.85)
    plt.axvline(0.0, linewidth=1.0)
    plt.title(title)
    plt.xlabel("denoised - raw")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


@torch.no_grad()
def random_robustness_sweep(
    feats: torch.Tensor,
    labels: torch.Tensor,
    forward_fn,
    eps_list,
    K: int = 64,
    out_dir: str = "./rand_robust",
    use_ball: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # correctness (for split)
    pred0 = forward_fn(feats).argmax(dim=1)
    correct = pred0.eq(labels)

    summary = {}
    for eps in eps_list:
        p_keep = random_robustness_prob(feats, forward_fn, eps=eps, K=K, use_ball=use_ball, chunk_K=K//10)
        summary[float(eps)] = {
            "p_keep_mean": float(p_keep.mean().item()),
            "p_keep_median": float(p_keep.median().item()),
            "p_keep_p10": float(torch.quantile(p_keep, 0.10).item()),
            "p_keep_p90": float(torch.quantile(p_keep, 0.90).item()),
        }

        plot_pkeep_hist(
            p_keep, correct,
            out_png=os.path.join(out_dir, f"pkeep_hist_eps_{eps:.4g}.png"),
            title=f"Random robustness: eps={eps} (K={K})",
        )

    return summary

def _hist_two(values: torch.Tensor, correct_mask: torch.Tensor, title: str, xlabel: str, out_path: str, bins: int = 60):
    v = _to_cpu(values).flatten()
    c = _to_cpu(correct_mask).flatten().bool()
    v_all = v
    v_cor = v[c]
    v_inc = v[~c]

    plt.figure(figsize=(7, 5))

    plt.hist(
        [v_cor.numpy(), v_inc.numpy()],
        bins=bins,
        alpha=0.45,
        color=["tab:blue", "tab:orange"],
        label=[f"class 1 (n={len(v_cor)})", f"class 0 (n={len(v_inc)})"],
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend()
    _savefig(out_path)

@torch.no_grad()
def random_robustness_sweep_dual_relative(
    feats: torch.Tensor,
    forward_fn,
    alpha_list,
    labels, 
    K: int = 128,
    out_dir: str = "./rand_robust_dual",
):
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for alpha in alpha_list:
        p_raw, p_den = random_robustness_prob_dual_relative(
            feats=feats, forward_fn=forward_fn, alpha=float(alpha), labels=labels, K=K, chunk_K=K//10 if K >= 512 else K
        )

        # two-hist plot
        plot_two_hist(
            p_raw, p_den,
            label_a="raw (fc(x))",
            label_b="denoised (fc(ode(x)))",
            title=f"Random robustness (relative): alpha={alpha}, K={K}",
            xlabel="P[pred stays same under random perturbations]",
            out_png=os.path.join(out_dir, f"pkeep_hist_alpha_{alpha:.4g}.png"),
        )
        
        _hist_two((p_den - p_raw).detach().cpu(), labels, f"Δ random robustness (relative): alpha={alpha}, K={K}", 'x', 
            os.path.join(out_dir, f"perclass-pkeep_delta_alpha_{alpha:.4g}.png"))
        
        # delta plot
        plot_delta_hist(
            p_den, p_raw,
            title=f"Δ random robustness (relative): alpha={alpha}, K={K}",
            out_png=os.path.join(out_dir, f"pkeep_delta_alpha_{alpha:.4g}.png"),
        )

        summary[float(alpha)] = {
            "p_keep_raw_mean": float(p_raw.mean().item()),
            "p_keep_den_mean": float(p_den.mean().item()),
            "p_keep_gain_mean": float((p_den - p_raw).mean().item()),
            "p_keep_raw_median": float(p_raw.median().item()),
            "p_keep_den_median": float(p_den.median().item()),
            "p_keep_gain_median": float((p_den - p_raw).median().item()),
        }

    return summary

def raddddddi_rand(model, phase, loader, device): 
    data: SodefAnalysisInputs = model.collect_feats(loader, device, return_outputs = True)
    
    n = 1500
    feats = data.before_feats[0:n]
    feats = feats.to(device)
    labels = data.labels[0:n]
    labels = labels.to(device)

    def forward_fn(x):
        x = x.to(device)
        z = model.ode_block(x)          # [N,64]
        denoised_logits = model.fc(z)   # [N,C]
        raw_logits = model.fc(x)
        return raw_logits, denoised_logits

    # eps_list = [10.0, 20.0]   # choose scale
    # summary = random_robustness_sweep(
    #     feats=feats,
    #     labels=labels,
    #     forward_fn=forward_fn,
    #     eps_list=eps_list,
    #     K=4096,
    #     out_dir="./rand_robust_before",
    #     use_ball=False,   # sphere (exact norm)
    # )
    # for k, v in summary.items():
    #     print('eps', k)
    #     print(v)

    alpha_list = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]  # 0.0, 0.01, 0.02, 0.05, 0.1, 0.2 # relative noise levels
    summary = random_robustness_sweep_dual_relative(
        feats=feats,
        forward_fn=forward_fn,
        alpha_list=alpha_list,
        labels=labels,
        K=4096,
        out_dir=f"./rand_robust_dual_relative_{phase}",
    )
    print(summary)
