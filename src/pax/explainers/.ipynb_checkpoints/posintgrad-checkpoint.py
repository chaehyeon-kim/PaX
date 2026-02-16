from __future__ import annotations

from typing import Callable, Optional

import torch

from .explainer_utils import (
    build_interpretable_z,
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


def intgrad(
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
    steps = int(kwargs.get("steps", 50))

    X_b, meta = normalize_input(X)
    if process_fn is None:
        process_fn = lambda x: x

    with torch.no_grad():
        base_logits = model(prepare_raw_input(X_b, meta, process_fn))
        target_idx = base_logits.argmax(dim=-1) if target is None else target

    B = X_b.shape[0]
    L = X_b.shape[2] if meta.shape_kind == "bcld" else X_b.shape[1]

    _, _, perm_idx = perturb_Xp(X_b, meta=meta, gate_mode="continuous", zp=torch.zeros(B, L))

    zf_grad = torch.zeros(B, L, device=X_b.device)
    zp_grad = torch.zeros(B, L, device=X_b.device)

    for s in range(1, steps + 1):
        alpha = float(s) / steps
        zf = torch.full((B, L), alpha, device=X_b.device, requires_grad=True)
        zp = torch.full((B, L), alpha, device=X_b.device, requires_grad=True)

        Xc, _ = perturb_X(X_b, meta=meta, gate_mode="continuous", zf=zf)
        Xp, _, _ = perturb_Xp(Xc, meta=meta, gate_mode="continuous", zp=zp, perm_idx=perm_idx)

        logits = model(prepare_raw_input(Xp, meta, process_fn))
        tlogits = _select_target_logit(logits, target_idx)
        loss = tlogits.sum()
        grad_zf, grad_zp = torch.autograd.grad(loss, (zf, zp), retain_graph=False, create_graph=False)
        zf_grad = zf_grad + grad_zf.detach()
        zp_grad = zp_grad + grad_zp.detach()

    feature_attr = zf_grad / steps
    if not position:
        return format_output(feature_attr, meta)

    position_attr = zp_grad / steps
    if position_pair:
        B = position_attr.shape[0]
        counterfactual_attr = torch.zeros(B, L, L, device=position_attr.device, dtype=position_attr.dtype)
        for b in range(B):
            for a in range(L):
                bdest = int(perm_idx[b, a].item())
                if bdest != a:
                    counterfactual_attr[b, a, bdest] = position_attr[b, a]
        return format_output((feature_attr, position_attr, counterfactual_attr), meta)
    importance = build_interpretable_z(feature_attr, position_attr, position=True)
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
    return intgrad(
        X,
        model,
        process_fn=process_fn,
        position=position,
        position_pair=position_pair,
        target=target,
        mask_rate=mask_rate,
        **kwargs,
    )