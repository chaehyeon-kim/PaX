from __future__ import annotations

from typing import Callable, Optional, Tuple

import itertools
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


def _shapley_value_weights(L: int, k: Tensor) -> Tensor:
    k_clamped = k.clamp(min=0, max=L - 1)
    log_w = torch.lgamma(k_clamped + 1) + torch.lgamma(torch.tensor(float(L), device=k.device) - k_clamped)
    log_w -= torch.lgamma(torch.tensor(float(L) + 1, device=k.device))
    return torch.exp(log_w)


def _pairwise_interaction_weights(L: int, k: Tensor) -> Tensor:
    k_clamped = k.clamp(min=0, max=L - 2)
    log_w = torch.lgamma(k_clamped + 1) + torch.lgamma(torch.tensor(float(L), device=k.device) - k_clamped - 1)
    log_w -= torch.lgamma(torch.tensor(float(L), device=k.device)) + torch.log(torch.tensor(2.0, device=k.device))
    return torch.exp(log_w)


def _counterfactual_to_perm(z_c: Tensor, L: int) -> Tensor:
    eye = torch.eye(L, device=z_c.device, dtype=torch.bool)
    counterfactual_mat = torch.zeros(z_c.shape[0], L, L, device=z_c.device, dtype=z_c.dtype)
    counterfactual_mat[:, ~eye] = z_c
    perm = counterfactual_mat.argmax(dim=-1)
    row_sum = counterfactual_mat.sum(dim=-1)
    base = torch.arange(L, device=z_c.device).unsqueeze(0).expand(z_c.shape[0], -1)
    perm = torch.where(row_sum > 0, perm, base)
    return perm.long()


def _eval_masks(
    X_b: Tensor,
    meta,
    masks: Tensor,
    position: bool,
    position_pair: bool,
    model: Callable[[Tensor], Tensor],
    process_fn: Callable[[Tensor], Tensor],
    yhat: int,
) -> Tensor:
    L = X_b.shape[2] if meta.shape_kind == "bcld" else X_b.shape[1]
    if position:
        zf = masks[:, :L]
        zp = masks[:, L : 2 * L]
        X0, _ = perturb_X(X_b, meta=meta, zf=zf, gate_mode="binary")
        if position_pair:
            z_c = masks[:, 2 * L :]
            perm_idx = _counterfactual_to_perm(z_c, L)
            keep = zp == 1
            base = torch.arange(L, device=zp.device).unsqueeze(0).expand_as(perm_idx)
            perm_idx = torch.where(keep, base, perm_idx)
            Xp0, _, _ = perturb_Xp(X0, meta=meta, zp=zp, perm_idx=perm_idx, gate_mode="binary")
        else:
            Xp0, _, _ = perturb_Xp(X0, meta=meta, zp=zp, gate_mode="binary")
        x_eval = Xp0
    else:
        zf = masks
        X0, _ = perturb_X(X_b, meta=meta, zf=zf, gate_mode="binary")
        x_eval = X0
    with torch.no_grad():
        logits = model(prepare_raw_input(x_eval, meta, process_fn))
        return _select_target_logit(logits, yhat)


def shap_iq(
    X: Tensor,
    model: Callable[[Tensor], Tensor],
    process_fn: Optional[Callable[[Tensor], Tensor]] = None,
    position: bool = False,
    position_pair: bool = False,
    target: Optional[int | Tensor] = None,
    mask_rate: float = 0.5,
    **kwargs,
) -> Tensor | Tuple[Tensor, Tensor]:
    if position_pair and not position:
        raise ValueError("position_pair=True requires position=True")
    n_masks = int(kwargs.get("n_masks", 256))
    max_pairs = int(kwargs.get("max_pairs", 256))
    topk_pairs = kwargs.get("topk_pairs", None)
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

    X_rep = X_b[:1].expand(n_masks, *X_b.shape[1:])
    X0, zf = perturb_X(X_rep, mask_rate=mask_rate, meta=meta, gate_mode="binary")
    if position:
        Xp0, zp, perm_idx = perturb_Xp(X0, mask_rate=mask_rate, meta=meta, gate_mode="binary")
        z_c = build_z_c(perm_idx) if position_pair else None
        masks = build_interpretable_z(zf, zp, z_c, position=True, position_pair=position_pair)
    else:
        masks = zf

    main = torch.zeros(B, F, device=X_b.device)
    inter = torch.zeros(B, F, F, device=X_b.device) if kwargs.get("return_interactions", False) else None

    for b in range(B):
        for feat in range(F):
            masks_feat = masks.clone()
            masks_feat[:, feat] = 0.0
            k = masks_feat.sum(dim=1)
            weights = _shapley_value_weights(F, k)
            f_s = _eval_masks(
                X_b[b : b + 1],
                meta,
                masks_feat,
                position,
                position_pair,
                model,
                process_fn,
                int(target_idx[b].item()),
            )
            masks_feat_i = masks_feat.clone()
            masks_feat_i[:, feat] = 1.0
            f_si = _eval_masks(
                X_b[b : b + 1],
                meta,
                masks_feat_i,
                position,
                position_pair,
                model,
                process_fn,
                int(target_idx[b].item()),
            )
            contrib = (weights * (f_si - f_s)).sum() / weights.sum().clamp_min(1e-6)
            main[b, feat] = contrib

        if inter is not None:
            all_pairs = list(itertools.combinations(range(F), 2))
            if topk_pairs is not None:
                scores = []
                for a, b2 in all_pairs:
                    scores.append((abs(main[b, a] * main[b, b2]).item(), (a, b2)))
                scores.sort(reverse=True)
                pairs = [p for _, p in scores[:topk_pairs]]
            elif len(all_pairs) > max_pairs:
                idx = torch.randperm(len(all_pairs), device=X_b.device)[:max_pairs].tolist()
                pairs = [all_pairs[j] for j in idx]
            else:
                pairs = all_pairs

            for a, b2 in pairs:
                masks_ab = masks.clone()
                masks_ab[:, a] = 0.0
                masks_ab[:, b2] = 0.0
                k = masks_ab.sum(dim=1)
                weights = _pairwise_interaction_weights(F, k)
                f_s = _eval_masks(
                    X_b[b : b + 1],
                    meta,
                    masks_ab,
                    position,
                    position_pair,
                    model,
                    process_fn,
                    int(target_idx[b].item()),
                )
                masks_a = masks_ab.clone(); masks_a[:, a] = 1.0
                f_sa = _eval_masks(
                    X_b[b : b + 1],
                    meta,
                    masks_a,
                    position,
                    position_pair,
                    model,
                    process_fn,
                    int(target_idx[b].item()),
                )
                masks_b = masks_ab.clone(); masks_b[:, b2] = 1.0
                f_sb = _eval_masks(
                    X_b[b : b + 1],
                    meta,
                    masks_b,
                    position,
                    position_pair,
                    model,
                    process_fn,
                    int(target_idx[b].item()),
                )
                masks_ab2 = masks_ab.clone(); masks_ab2[:, a] = 1.0; masks_ab2[:, b2] = 1.0
                f_sab = _eval_masks(
                    X_b[b : b + 1],
                    meta,
                    masks_ab2,
                    position,
                    position_pair,
                    model,
                    process_fn,
                    int(target_idx[b].item()),
                )
                delta = f_sab - f_sa - f_sb + f_s
                contrib = (weights * delta).sum() / weights.sum().clamp_min(1e-6)
                inter[b, a, b2] = contrib
                inter[b, b2, a] = contrib

    if position and position_pair:
        feature_attr = main[:, :L]
        position_attr = main[:, L : 2 * L]
        counterfactual_flat = main[:, 2 * L :]
        eye = torch.eye(L, device=main.device, dtype=torch.bool)
        counterfactual_attr = torch.zeros(B, L, L, device=main.device, dtype=main.dtype)
        counterfactual_attr[:, ~eye] = counterfactual_flat
        main_out = format_output((feature_attr, position_attr, counterfactual_attr), meta)
    else:
        main_out = format_output(main, meta)
    if inter is not None:
        return main_out, format_output(inter, meta)
    return main_out


def explain(
    X: Tensor,
    model: Callable[[Tensor], Tensor],
    process_fn: Optional[Callable[[Tensor], Tensor]] = None,
    position: bool = False,
    position_pair: bool = False,
    target: Optional[int | Tensor] = None,
    mask_rate: float = 0.5,
    **kwargs,
) -> Tensor | Tuple[Tensor, Tensor]:
    return shap_iq(
        X,
        model,
        process_fn=process_fn,
        position=position,
        position_pair=position_pair,
        target=target,
        mask_rate=mask_rate,
        **kwargs,
    )