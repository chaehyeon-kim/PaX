from __future__ import annotations

from typing import Callable, Optional

import torch

from .explainer_utils import (
    _get_L,
    format_output,
    lime_kernel,
    normalize_input,
    offdiag_record,
    permute_content,
    perturb_X,
    pos_swap,
    prepare_raw_input,
    unflatten_offdiag,
    weighted_ridge,
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


def _batched_target_logits(model, X_eval, meta, process_fn, target: int, batch_size: int) -> Tensor:
    outs = []
    with torch.no_grad():
        for s in range(0, X_eval.shape[0], batch_size):
            logits = model(prepare_raw_input(X_eval[s : s + batch_size], meta, process_fn))
            outs.append(_select_target_logit(logits, target))
    return torch.cat(outs)


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
    """PaX-LIME (Section 6.2, Algorithms 2-4).

    LIME runs verbatim on the reference-vector coordinates z' = (zx, zp), extended by the
    off-diagonal assignment record R when position_pair=True:
      1. sample a binary mask z' with P(z'_k = 1) = mask_rate (0.5 is Unif{0,1}^{2n}, Algorithm 3),
      2. evaluate one draw of g': content masking on zx, PosSwap on zp. This is a single-sample
         unbiased estimate of v(z'), and the drawn sigma is the only new randomness,
      3. weight the sample by the LIME proximity kernel pi(z') on the sampled mask. R is induced
         by the sampler rather than sampled by the explainer, so it does not enter pi,
      4. fit the weighted ridge surrogate on (zx, zp[, offdiag R]).

    kwargs: n_masks (default 100), l2 (default 1.0), kernel_width (default 0.25),
    batch_size (default 64), seed.

    Returns, per input:
      position=False                       -> alpha_x [L]
      position=True, position_pair=False   -> cat(alpha_x, alpha_p) [2L]
      position_pair=True                   -> (alpha_x [L], alpha_p [L], alpha_q [L, L])
    alpha_q[i, j] scores content i occupying slot j (zero diagonal). Compare coefficients
    within a block only, since cross-block magnitudes depend on the regularization.
    """
    if position_pair and not position:
        raise ValueError("position_pair=True requires position=True")
    n_masks = int(kwargs.get("n_masks", 100))
    l2 = float(kwargs.get("l2", 1.0))
    kernel_width = float(kwargs.get("kernel_width", 0.25))
    batch_size = int(kwargs.get("batch_size", 64))
    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))

    X_b, meta = normalize_input(X)
    if process_fn is None:
        process_fn = lambda x: x

    with torch.no_grad():
        base_logits = model(prepare_raw_input(X_b, meta, process_fn))
    if target is None:
        target_idx = base_logits.argmax(dim=-1)
    elif isinstance(target, int):
        target_idx = torch.full((X_b.shape[0],), target, dtype=torch.long)
    else:
        target_idx = target.view(-1)

    B = X_b.shape[0]
    L = _get_L(X_b, meta)
    device = X_b.device
    F = L if not position else 2 * L + (L * (L - 1) if position_pair else 0)
    importance = torch.zeros(B, F, device=device)

    for b in range(B):
        X_rep = X_b[b : b + 1]
        zf = torch.bernoulli(torch.full((n_masks, L), mask_rate, device=device))
        X0, _ = perturb_X(X_rep, meta=meta, zf=zf)  # content perturbation g(x, zx)
        if position:
            zp = torch.bernoulli(torch.full((n_masks, L), mask_rate, device=device))
            draw = pos_swap(zp)  # pool = {i : zp_i = 0}, sigma ~ Unif(Sym(pool))
            x_eval = permute_content(X0, draw.src, meta)
            mask = torch.cat([zf, zp], dim=1)
            z = torch.cat([mask, offdiag_record(draw.dest)], dim=1) if position_pair else mask
        else:
            x_eval, mask, z = X0, zf, zf

        y = _batched_target_logits(model, x_eval, meta, process_fn, int(target_idx[b].item()), batch_size)
        w = lime_kernel(mask, kernel_width)
        importance[b] = weighted_ridge(z, y, w, l2)

    if position and position_pair:
        feature_attr = importance[:, :L]
        position_attr = importance[:, L : 2 * L]
        counterfactual_attr = unflatten_offdiag(importance[:, 2 * L :], L)
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
