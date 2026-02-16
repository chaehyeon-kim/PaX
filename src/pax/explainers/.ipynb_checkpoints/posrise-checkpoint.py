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


def rise(
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
            scores = _select_target_logit(logits, int(target_idx[b].item()))

        numer = (scores.view(-1, 1) * z).sum(dim=0)
        p = z.mean().clamp_min(1e-6)
        importance[b] = numer / (n_masks * p)

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
    return rise(
        X,
        model,
        process_fn=process_fn,
        position=position,
        position_pair=position_pair,
        target=target,
        mask_rate=mask_rate,
        **kwargs,
    )