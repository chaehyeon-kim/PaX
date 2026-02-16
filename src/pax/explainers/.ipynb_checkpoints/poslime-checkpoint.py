from __future__ import annotations

from typing import Callable, Optional

import torch

from .explainer_utils import (
    RidgeRegression,
    build_interpretable_z,
    build_z_c,
    format_output,
    normalize_input,
    perturb_X,
    perturb_Xp,
    prepare_raw_input,
    weight,
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


def lime(
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
    n_masks = int(kwargs.get("n_masks", 100))
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

    X_lime_list, y_lime_list, weights_list = [], [], []
    for _ in range(n_masks):
        X0, zf = perturb_X(X_b, mask_rate=mask_rate, meta=meta, gate_mode="binary")
        if position:
            Xp0, zp, perm_idx = perturb_Xp(X0, mask_rate=mask_rate, meta=meta, gate_mode="binary")
            z_c = build_z_c(perm_idx) if position_pair else None
            z = build_interpretable_z(zf, zp, z_c, position=True, position_pair=position_pair)
            X_weights = weight(X0, X_b, meta=meta)
            Xp_weights = weight(Xp0, X0, meta=meta)
            if position_pair:
                counterfactual_weights = torch.ones_like(z_c)
                weights_sample = torch.cat([X_weights, Xp_weights, counterfactual_weights], dim=1)
            else:
                weights_sample = torch.cat([X_weights, Xp_weights], dim=1)
            x_eval = Xp0
        else:
            z = zf
            weights_sample = weight(X0, X_b, meta=meta)
            x_eval = X0

        with torch.no_grad():
            outputs0 = model(prepare_raw_input(x_eval, meta, process_fn))
            logit0 = _select_target_logit(outputs0, target_idx)

        X_lime_list.append(z.unsqueeze(1))
        y_lime_list.append(logit0.unsqueeze(1))
        weights_list.append(weights_sample.unsqueeze(1))

    X_lime = torch.cat(X_lime_list, dim=1)
    y_lime = torch.cat(y_lime_list, dim=1)
    weights = torch.cat(weights_list, dim=1)

    importances = []
    for b in range(X_b.shape[0]):
        r_model = RidgeRegression(X_lime.shape[2], alpha=l2).to(X_b.device)
        optimizer = torch.optim.Adam(r_model.parameters(), lr=0.01)
        for _ in range(100):
            r_model.train()
            optimizer.zero_grad()
            loss = r_model.compute_loss(X_lime[b], y_lime[b], weights[b])
            loss.backward(retain_graph=True)
            optimizer.step()
        w = r_model.linear.weight.detach().view(-1)
        importances.append(w)

    importance = torch.stack(importances, dim=0)
    if position and position_pair:
        L = X_b.shape[2] if meta.shape_kind == "bcld" else X_b.shape[1]
        feature_attr = importance[:, :L]
        position_attr = importance[:, L : 2 * L]
        counterfactual_flat = importance[:, 2 * L :]
        eye = torch.eye(L, device=importance.device, dtype=torch.bool)
        counterfactual_attr = torch.zeros(
            importance.shape[0],
            L,
            L,
            device=importance.device,
            dtype=importance.dtype,
        )
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
    return lime(
        X,
        model,
        process_fn=process_fn,
        position=position,
        position_pair=position_pair,
        target=target,
        mask_rate=mask_rate,
        **kwargs,
    )