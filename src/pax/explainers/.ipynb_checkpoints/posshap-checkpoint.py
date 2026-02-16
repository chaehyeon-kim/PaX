from __future__ import annotations

from typing import Callable, Optional

import torch

from .explainer_utils import (
    build_interpretable_z,
    build_z_c,
    format_output,
    normalize_input,
    perturb_X,
    perturb_Xp,
    prepare_raw_input,
)

Tensor = torch.Tensor

def _select_target_logit(logits: Tensor, target: Optional[int | Tensor]) -> Tensor:
    if target is None:
        target_idx = logits.argmax(dim=-1)
    elif isinstance(target, int):
        target_idx = torch.full((logits.shape[0],), target, device=logits.device, dtype=torch.long)
    else:
        target_idx = target.to(logits.device)
    return logits.gather(1, target_idx.view(-1, 1)).squeeze(1)


def _shap_kernel_weights(masks: Tensor) -> Tensor:
    L = masks.shape[1]
    k = masks.sum(dim=1)
    k_clamped = k.clamp(min=1, max=L - 1)
    log_comb = torch.lgamma(torch.tensor(float(L) + 1, device=masks.device)) - (
        torch.lgamma(k_clamped + 1) + torch.lgamma(torch.tensor(float(L), device=masks.device) - k_clamped + 1)
    )
    comb = torch.exp(log_comb)
    weights = (L - 1) / (comb * k_clamped * (L - k_clamped))
    weights[(k == 0) | (k == L)] = 0.0
    return weights


def _weighted_ridge_with_intercept(X: Tensor, y: Tensor, w: Tensor, l2: float) -> Tensor:
    y = y.view(-1)
    w = w.view(-1, 1)
    ones = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)
    Xc = torch.cat([ones, X], dim=1)
    Xw = Xc * w
    R = torch.eye(Xc.shape[1], device=X.device, dtype=X.dtype)
    R[0, 0] = 0.0
    A = Xw.t() @ Xc + l2 * R
    b = Xw.t() @ y
    sol = torch.linalg.solve(A, b)
    return sol[1:]


def shap(
    X: Tensor,
    model: Callable[[Tensor], Tensor],
    process_fn: Optional[Callable[[Tensor], Tensor]] = None,
    position: bool = False,
    position_pair: bool = False,
    target: Optional[int | Tensor] = None,
    mask_rate: float = 0.5,
    **kwargs,
) -> Tensor:
    if position_pair and not position:
        raise ValueError("position_pair=True requires position=True")
    n_masks = int(kwargs.get("n_masks", 512))
    l2 = float(kwargs.get("l2", 1.0))
    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))

    X_b, meta = normalize_input(X)
    if process_fn is None:
        process_fn = lambda x: x

    with torch.no_grad():
        base_logits = model(prepare_raw_input(X_b, meta, process_fn))
        target_idx = base_logits.argmax(dim=-1) if target is None else target

    B = X_b.shape[0]
    L = X_b.shape[2] if meta.shape_kind == "bcld" else X_b.shape[1]
    if position:
        F = 2 * L + (L * (L - 1) if position_pair else 0)
    else:
        F = L
    importance = torch.zeros(B, F, device=X_b.device)

    for b in range(B):
        X_rep = X_b[b : b + 1].expand(n_masks, *X_b.shape[1:])
        X0, zf = perturb_X(X_rep, mask_rate=mask_rate, meta=meta, gate_mode="binary")
        if position:
            Xp0, zp, perm_idx = perturb_Xp(X0, mask_rate=mask_rate, meta=meta, gate_mode="binary")
            z_c = build_z_c(perm_idx) if position_pair else None
            z = build_interpretable_z(zf, zp, z_c, position=True, position_pair=position_pair)
            x_eval = Xp0
        else:
            z = zf
            x_eval = X0

        with torch.no_grad():
            logits = model(prepare_raw_input(x_eval, meta, process_fn))
            tlogits = _select_target_logit(logits, int(target_idx[b].item()))

        weights = _shap_kernel_weights(z)
        phi = _weighted_ridge_with_intercept(z, tlogits, weights, l2=l2)
        importance[b] = phi

    if position and position_pair:
        feature_attr = importance[:, :L]
        position_attr = importance[:, L : 2 * L]
        counterfactual_flat = importance[:, 2 * L :]
        eye = torch.eye(L, device=importance.device, dtype=torch.bool)
        counterfactual_attr = torch.zeros(B, L, L, device=importance.device, dtype=importance.dtype)
        counterfactual_attr[:, ~eye] = counterfactual_flat
        return format_output((feature_attr, position_attr, counterfactual_attr), meta)
    return format_output(importance, meta)


def explain(
    X: Tensor,
    model: Callable[[Tensor], Tensor],
    process_fn: Optional[Callable[[Tensor], Tensor]] = None,
    position: bool = False,
    position_pair: bool = False,
    target: Optional[int | Tensor] = None,
    mask_rate: float = 0.5,
    **kwargs,
) -> Tensor:
    return shap(
        X,
        model,
        process_fn=process_fn,
        position=position,
        position_pair=position_pair,
        target=target,
        mask_rate=mask_rate,
        **kwargs,
    )