from __future__ import annotations

import warnings
from typing import Callable, Optional

import torch

from .explainer_utils import (
    _get_L,
    format_output,
    normalize_input,
    permute_content,
    perturb_X,
    pos_swap,
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


def _binned_contrasts(y: Tensor, coins: Tensor, dest: Tensor):
    """Eq. (9) at one grid level, from M sampler draws.

    y [M] outputs, coins [M, L] (b_i = 1 means i is in the pool), dest [M, L] (sigma(i)).
    Returns the positional partials (Lemma 1), the pair contrasts (Definition 5), and masks
    marking which entries had non-empty bins.
    """
    M, L = coins.shape
    y = y.double()
    stay = (~coins).double()
    moved = coins.double()
    n0, n1 = stay.sum(0), moved.sum(0)
    mean0 = (stay * y[:, None]).sum(0) / n0.clamp_min(1)
    mean1 = (moved * y[:, None]).sum(0) / n1.clamp_min(1)
    pos_valid = (n0 > 0) & (n1 > 0)
    d_pos = mean0 - mean1  # y[b_i = 0] - y[b_i = 1]

    cell = (torch.arange(L, device=dest.device) * L).unsqueeze(0) + dest  # flat index of (i, sigma(i))
    nR = torch.zeros(L * L, dtype=torch.float64, device=dest.device)
    sR = torch.zeros(L * L, dtype=torch.float64, device=dest.device)
    nR.scatter_add_(0, cell.reshape(-1), torch.ones(M * L, dtype=torch.float64, device=dest.device))
    sR.scatter_add_(0, cell.reshape(-1), y[:, None].expand(M, L).reshape(-1))
    nR, sR = nR.view(L, L), sR.view(L, L)
    offdiag = ~torch.eye(L, dtype=torch.bool, device=dest.device)
    cf_valid = (nR > 0) & (n0 > 0)[:, None] & offdiag
    c_pair = sR / nR.clamp_min(1) - mean0[:, None]  # y[R_ij = 1] - y[b_i = 0]
    return d_pos, pos_valid, c_pair, cf_valid


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
    """PaX-IntGrad (Section 5, Algorithm 10).

    Baseline z' = 0 (content at the base IG baseline, positions fully shuffled with
    sigma ~ Unif(Sym([n]))), path z'(t) = t 1, and derivatives:
      - content: native IG derivative d f' / d zx_i via autograd, averaged over sampler draws,
      - positions: exact partial of Lemma 1, y[b_i = 0] - y[b_i = 1], from forward passes only,
      - pairs: oracle of Definition 5, y[R_ij = 1] - y[b_i = 0], from the same draws.
    The model only ever sees hard permutations of the input. Positional embeddings are never
    interpolated.

    With position=False this reduces to plain IG on content (positions retained, zp = 1).
    mask_rate is unused and kept for a uniform explainer signature.

    kwargs: steps (K grid levels, default 50), n_draws (M draws per level, default 64),
    batch_size (default 64), seed. The grid is the midpoint rule t_k = (k - 1/2) / K, which
    avoids t = 1 (no draw has b_i = 1) and t = 0 (no draw has b_i = 0).

    A (position, level) cell or a (pair, level) cell whose bin is empty for the given draws is
    left out of that coordinate's quadrature average. A warning reports how often this happens.
    Pairs with no non-empty level are returned as 0.

    Returns, per input:
      position=False                       -> alpha_x [L]
      position=True, position_pair=False   -> cat(alpha_x, alpha_p) [2L]
      position_pair=True                   -> (alpha_x [L], alpha_p [L], alpha_q [L, L])
    alpha_q[i, j] scores content i occupying slot j (zero diagonal).
    """
    if position_pair and not position:
        raise ValueError("position_pair=True requires position=True")
    steps = int(kwargs.get("steps", 50))
    n_draws = int(kwargs.get("n_draws", 64))
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
    M = n_draws if position else 1  # without position perturbation f' is deterministic
    ts = ((torch.arange(steps, dtype=torch.float64) + 0.5) / steps).tolist()

    feature_attr = torch.zeros(B, L, device=device)
    position_attr = torch.zeros(B, L, device=device)
    counterfactual_attr = torch.zeros(B, L, L, device=device)
    skipped_pos_cells, unobserved_pairs = 0.0, 0.0

    with torch.enable_grad():
        for b in range(B):
            X_rep = X_b[b : b + 1]
            tgt = int(target_idx[b].item())
            grad_sum = torch.zeros(L, dtype=torch.float64, device=device)
            pos_sum = torch.zeros(L, dtype=torch.float64, device=device)
            pos_n = torch.zeros(L, dtype=torch.float64, device=device)
            cf_sum = torch.zeros(L, L, dtype=torch.float64, device=device)
            cf_n = torch.zeros(L, L, dtype=torch.float64, device=device)

            for t in ts:
                ys, coins, dests = [], [], []
                for s in range(0, M, batch_size):
                    m = min(batch_size, M - s)
                    zf = torch.full((m, L), t, device=device, requires_grad=True)
                    X0, _ = perturb_X(X_rep, meta=meta, zf=zf)  # g(x, t 1)
                    if position:
                        draw = pos_swap(torch.full((m, L), t, device=device))  # PosSwap(P, t 1)
                        x_eval = permute_content(X0, draw.src, meta)
                    else:
                        x_eval = X0
                    y = _select_target_logit(model(prepare_raw_input(x_eval, meta, process_fn)), tgt)
                    (g,) = torch.autograd.grad(y.sum(), zf)
                    grad_sum += g.detach().double().sum(0)
                    if position:
                        ys.append(y.detach())
                        coins.append(draw.coins)
                        dests.append(draw.dest)

                if position:
                    d_pos, pos_valid, c_pair, cf_valid = _binned_contrasts(
                        torch.cat(ys), torch.cat(coins), torch.cat(dests)
                    )
                    pos_sum += torch.where(pos_valid, d_pos, torch.zeros_like(d_pos))
                    pos_n += pos_valid.double()
                    cf_sum += torch.where(cf_valid, c_pair, torch.zeros_like(c_pair))
                    cf_n += cf_valid.double()

            # Coordinate difference along the path is 1 - 0 = 1 for every coordinate.
            feature_attr[b] = (grad_sum / (steps * M)).to(feature_attr.dtype)
            if position:
                position_attr[b] = (pos_sum / pos_n.clamp_min(1)).to(position_attr.dtype)
                counterfactual_attr[b] = (cf_sum / cf_n.clamp_min(1)).to(counterfactual_attr.dtype)
                skipped_pos_cells += float((steps - pos_n).sum().item())
                offdiag = ~torch.eye(L, dtype=torch.bool, device=device)
                unobserved_pairs += float((cf_n[offdiag] == 0).sum().item())

    if position:
        frac_pos = skipped_pos_cells / (B * L * steps)
        frac_pair = unobserved_pairs / max(B * L * (L - 1), 1)
        if frac_pos > 0 or (position_pair and frac_pair > 0):
            msg = f"PaX-IntGrad: {frac_pos:.1%} of (position, level) cells had an empty bin"
            if position_pair:
                msg += f", {frac_pair:.1%} of pairs (i, j) were never observed and are set to 0"
            warnings.warn(msg + ". Increase n_draws to reduce this.")

    if not position:
        return format_output(feature_attr, meta)
    if position_pair:
        return format_output((feature_attr, position_attr, counterfactual_attr), meta)
    return format_output(torch.cat([feature_attr, position_attr], dim=1), meta)


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
