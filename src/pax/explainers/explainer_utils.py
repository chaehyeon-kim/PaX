from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


Tensor = torch.Tensor


@dataclass
class InputMeta:
    shape_kind: str
    squeeze: bool


def normalize_input(X: Tensor) -> Tuple[Tensor, InputMeta]:
    """Normalize input to batched tensor with meta.

    Supported shapes: (L,D), (B,L,D), (B,C,L,D).
    Note: (C,L,D) is not supported; use (B,C,L,D) instead.
    """
    if X.dim() == 2:
        return X.unsqueeze(0), InputMeta("bld", True)
    if X.dim() == 3:
        return X, InputMeta("bld", False)
    if X.dim() == 4:
        return X, InputMeta("bcld", False)
    raise ValueError("Unsupported input shape")


def prepare_raw_input(X_b: Tensor, meta: InputMeta, process_fn: Optional[Callable[[Tensor], Tensor]]) -> Tensor:
    return process_fn(X_b) if process_fn is not None else X_b


def format_output(out: Tensor | tuple[Tensor, Tensor, Tensor], meta: InputMeta) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    if isinstance(out, tuple):
        f, p, cf = out
        if meta.squeeze:
            return f.squeeze(0), p.squeeze(0), cf.squeeze(0)
        return out
    return out.squeeze(0) if meta.squeeze else out


def build_interpretable_z(
    zf: Tensor,
    zp: Optional[Tensor] = None,
    z_c: Optional[Tensor] = None,
    position: bool = False,
    position_pair: bool = False,
) -> Tensor:
    if not position:
        return zf
    if zp is None:
        raise ValueError("zp required when position=True")
    if position_pair:
        if z_c is None:
            raise ValueError("z_c required when position_pair=True")
        return torch.cat([zf, zp, z_c], dim=1)
    return torch.cat([zf, zp], dim=1)


def build_z_c(perm: Tensor) -> Tensor:
    """Build directed counterfactual indicator from perm mapping.

    Input: perm [B, L]
    Output: z_c [B, L*(L-1)], row-major without diagonal.
    """
    B, L = perm.shape
    one_hot = torch.nn.functional.one_hot(perm, num_classes=L).float()
    eye = torch.eye(L, device=perm.device).bool()
    mask = ~eye
    z_c = one_hot[:, mask].view(B, L * (L - 1))
    return z_c


def weight(sample: Tensor, original: Tensor, sigma: float = 1.0, meta: Optional[InputMeta] = None) -> Tensor:
    """RBF kernel weights per feature (patch/token)."""
    if meta is not None and meta.shape_kind == "bcld":
        diff = sample - original
        distances = diff.pow(2).sum(dim=1).sum(dim=-1).sqrt()
    else:
        distances = torch.norm(sample - original, dim=-1)
    return torch.exp(-(distances ** 2) / (2 * sigma ** 2))


class RidgeRegression(torch.nn.Module):
    def __init__(self, num_features: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1, bias=True)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def compute_loss(self, x: Tensor, y: Tensor, sample_weights: Tensor) -> Tensor:
        weighted_x = x * sample_weights
        predictions = self.forward(weighted_x.unsqueeze(0))
        diff = predictions.squeeze() - y
        mse_loss = torch.mean(diff ** 2)
        ridge_loss = self.alpha * torch.sum(self.linear.weight ** 2)
        return mse_loss + ridge_loss


def _get_L(X_b: Tensor, meta: InputMeta) -> int:
    return X_b.shape[2] if meta.shape_kind == "bcld" else X_b.shape[1]


def _expand_gate(gate: Tensor, X_b: Tensor, meta: InputMeta) -> Tensor:
    if meta.shape_kind == "bcld":
        return gate.unsqueeze(1).unsqueeze(-1)
    return gate.unsqueeze(-1)


def _sample_gate(B: int, L: int, gate_mode: str, mask_rate: float, device: torch.device) -> Tensor:
    if gate_mode == "binary":
        return torch.bernoulli(torch.full((B, L), mask_rate, device=device))
    if gate_mode == "continuous":
        return torch.rand(B, L, device=device)
    raise ValueError("gate_mode must be 'binary' or 'continuous'")


def _permute_masked(masked_idx: Tensor, L: int, device: torch.device) -> Tensor:
    idx = torch.arange(L, device=device)
    m = masked_idx.numel()
    if m == 0:
        return idx
    if m == 1:
        return idx
    perm = masked_idx[torch.randperm(m, device=device)]
    if (perm == masked_idx).any():
        perm = torch.roll(masked_idx, shifts=1)
    idx[masked_idx] = perm
    return idx


def _permute_full(L: int, device: torch.device) -> Tensor:
    perm = torch.randperm(L, device=device)
    if (perm == torch.arange(L, device=device)).any():
        perm = torch.roll(torch.arange(L, device=device), shifts=1)
    return perm


def perturb_X(
    X_b: Tensor,
    mask_rate: float = 0.5,
    meta: InputMeta | None = None,
    gate_mode: str = "binary",
    zf: Optional[Tensor] = None,
    alpha: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """Feature perturbation: f'_i = zf_i * f_i."""
    if meta is None:
        X_b, meta = normalize_input(X_b)
    B = X_b.shape[0]
    L = _get_L(X_b, meta)
    if zf is None:
        if alpha is not None:
            zf = torch.full((B, L), float(alpha), device=X_b.device)
        else:
            zf = _sample_gate(B, L, gate_mode, mask_rate, X_b.device)
    if zf.dim() == 1:
        zf = zf.unsqueeze(0)
    gate = _expand_gate(zf, X_b, meta)
    if zf.shape[0] == X_b.shape[0]:
        X0 = X_b * gate
    else:
        if X_b.shape[0] != 1:
            raise ValueError("Gate batch must match X or X must have batch=1.")
        X_rep = X_b.expand(zf.shape[0], *X_b.shape[1:])
        X0 = X_rep * gate
    return X0, zf


def _gather_permuted(Xc: Tensor, perm_idx: Tensor, meta: InputMeta) -> Tensor:
    if meta.shape_kind == "bcld":
        B, C, L, D = Xc.shape
        idx = perm_idx.view(B, 1, L, 1).expand(B, C, L, D)
        return torch.gather(Xc, 2, idx)
    B, L, D = Xc.shape
    idx = perm_idx.view(B, L, 1).expand(B, L, D)
    return torch.gather(Xc, 1, idx)


def apply_position_indicator(
    Xf: Tensor,
    zp: Tensor,
    meta: InputMeta,
    gate_mode: str = "binary",
    perm_idx: Optional[Tensor] = None,
) -> Tensor:
    """Apply position gating by permuting feature embeddings.

    Binary: permute only indices where zp==0.
    Continuous: use a fixed perm_idx and interpolate with zp.
    """
    if zp.dim() == 1:
        zp = zp.unsqueeze(0)
    B = zp.shape[0]
    L = _get_L(Xf, meta)

    if Xf.shape[0] == 1 and B > 1:
        Xf = Xf.expand(B, *Xf.shape[1:])
    if Xf.shape[0] != B:
        raise ValueError("Gate batch must match X or X must have batch=1.")

    if perm_idx is None:
        perm_idx = torch.arange(L, device=Xf.device).unsqueeze(0).expand(B, -1).clone()
        if gate_mode == "binary":
            for b in range(B):
                masked_idx = torch.nonzero(zp[b] == 0, as_tuple=False).view(-1)
                perm_idx[b] = _permute_masked(masked_idx, L, Xf.device)
        else:
            for b in range(B):
                perm_idx[b] = _permute_full(L, Xf.device)

    Xperm = _gather_permuted(Xf, perm_idx, meta)

    gate = _expand_gate(zp, Xf, meta)
    return gate * Xf + (1.0 - gate) * Xperm


def perturb_Xp(
    X0: Tensor,
    mask_rate: float = 0.5,
    meta: InputMeta | None = None,
    gate_mode: str = "binary",
    zp: Optional[Tensor] = None,
    alpha: Optional[float] = None,
    perm_idx: Optional[Tensor] = None,
    max_resamples: int = 20,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Position perturbation via feature permutation.

    Binary: permute only indices where indicator==0 (resample if needed).
    Continuous: interpolate with zp and fixed perm_idx.
    """
    if meta is None:
        X0, meta = normalize_input(X0)
    B = X0.shape[0]
    L = _get_L(X0, meta)

    sampled = False
    if zp is None:
        if alpha is not None:
            zp = torch.full((B, L), float(alpha), device=X0.device)
        else:
            zp = _sample_gate(B, L, gate_mode, mask_rate, X0.device)
            sampled = True

    if gate_mode == "binary" and sampled:
        for b in range(B):
            zeros = int((zp[b] == 0).sum().item())
            if zeros == 0:
                continue
            if zeros == 1:
                resamples = 0
                while resamples < max_resamples:
                    cand = _sample_gate(1, L, gate_mode, mask_rate, X0.device).squeeze(0)
                    zeros_c = int((cand == 0).sum().item())
                    if zeros_c >= 2:
                        zp[b] = cand
                        break
                    resamples += 1
                else:
                    zp[b] = torch.ones(L, device=X0.device)

    if perm_idx is None:
        perm_idx = torch.arange(L, device=X0.device).unsqueeze(0).expand(B, -1).clone()
        if gate_mode == "binary":
            for b in range(B):
                masked_idx = torch.nonzero(zp[b] == 0, as_tuple=False).view(-1)
                perm_idx[b] = _permute_masked(masked_idx, L, X0.device)
        else:
            for b in range(B):
                perm_idx[b] = _permute_full(L, X0.device)
    Xp0 = apply_position_indicator(X0, zp, meta, gate_mode=gate_mode, perm_idx=perm_idx)
    return Xp0, zp, perm_idx

# ---------------------------------------------------------------------------
# PosSwap sampler and assignment record (PaX formalism, Definitions 2 and 4).
#
# Index conventions used by everything below:
#   src[n, j]  = content placed at slot j       (sigma^{-1}(j))
#   dest[n, i] = slot that content i moves to   (sigma(i))
#   R[n, i, j] = 1[sigma(i) = j]                (content i occupies slot j)
# ---------------------------------------------------------------------------


@dataclass
class PosSwapDraw:
    src: Tensor
    dest: Tensor
    coins: Tensor


def pos_swap(zp: Tensor) -> PosSwapDraw:
    """PosSwap (Definition 2, Algorithm 1).

    zp is an [N, L] retention vector in [0, 1] (1 retains position i, 0 fully perturbs it).
    Draws coins b_i ~ Bern(1 - zp_i) independently, forms the pool S = {i : b_i = 1},
    and draws sigma uniformly from Sym(S), extended by the identity on the complement.
    Fixed points inside the pool are kept, and |S| <= 1 yields the identity.
    At binary zp the coins are degenerate and the pool equals the mask's perturbed set.
    """
    if zp.dim() == 1:
        zp = zp.unsqueeze(0)
    N, L = zp.shape
    device = zp.device
    coins = torch.rand(N, L, device=device) < (1.0 - zp)
    ar = torch.arange(L, device=device).expand(N, L)
    # Pool slots in ascending order, followed by the fixed slots.
    slot_order = torch.argsort(torch.where(coins, ar, ar + L), dim=1)
    # Pool contents in uniformly random order, followed by the fixed contents.
    content_keys = torch.where(coins, torch.rand(N, L, device=device), (ar + L).to(torch.float32))
    content_order = torch.argsort(content_keys, dim=1)
    src = torch.empty(N, L, dtype=torch.long, device=device).scatter_(1, slot_order, content_order)
    dest = torch.empty(N, L, dtype=torch.long, device=device).scatter_(1, content_order, slot_order)
    return PosSwapDraw(src=src, dest=dest, coins=coins)


def permute_content(X0: Tensor, src: Tensor, meta: InputMeta) -> Tensor:
    """Evaluate f'(., Pi_sigma P) by reordering the input (Definition 1, eq. 2).

    Slot j receives content src[:, j], so content i lands at slot sigma(i).
    """
    return _gather_permuted(X0, src, meta)


def offdiag_record(dest: Tensor) -> Tensor:
    """Off-diagonal entries of the assignment record R (Definition 4).

    Returns [N, L*(L-1)], row-major over (i, j != i), with entry 1 iff sigma(i) = j.
    """
    L = dest.shape[1]
    R = torch.nn.functional.one_hot(dest, num_classes=L).to(torch.float32)
    offdiag = ~torch.eye(L, dtype=torch.bool, device=dest.device)
    return R[:, offdiag]


def unflatten_offdiag(flat: Tensor, L: int) -> Tensor:
    """Inverse of the off-diagonal flattening: [B, L*(L-1)] -> [B, L, L] with a zero diagonal."""
    out = torch.zeros(flat.shape[0], L, L, device=flat.device, dtype=flat.dtype)
    out[:, ~torch.eye(L, dtype=torch.bool, device=flat.device)] = flat
    return out


def lime_kernel(mask: Tensor, kernel_width: float = 0.25) -> Tensor:
    """LIME proximity kernel pi(z) on a binary mask, one weight per sample.

    Matches the reference LIME image/text explainers: cosine distance d between z and the
    all-ones (unperturbed) mask, weight sqrt(exp(-d^2 / kernel_width^2)).
    """
    cos = torch.nn.functional.cosine_similarity(mask, torch.ones_like(mask), dim=1, eps=1e-12)
    d = 1.0 - cos
    return torch.sqrt(torch.exp(-(d ** 2) / kernel_width ** 2))


def weighted_ridge(Z: Tensor, y: Tensor, w: Tensor, l2: float) -> Tensor:
    """argmin_beta sum_m w_m (y_m - beta_0 - beta^T z_m)^2 + l2 * ||beta||^2.

    The intercept is unpenalized and not returned. Solved in closed form, using the dual
    (M x M) system when features outnumber samples, e.g. with the L*(L-1) pair coordinates.
    """
    out_dtype = Z.dtype
    Z, y, w = Z.double(), y.double().view(-1), w.double().view(-1)
    wsum = w.sum()
    Zc = Z - (w[:, None] * Z).sum(0) / wsum
    yc = y - (w * y).sum() / wsum
    sw = w.sqrt()
    A = sw[:, None] * Zc
    c = sw * yc
    M, F = A.shape
    if l2 <= 0:
        beta = torch.linalg.pinv(A) @ c
    elif F <= M:
        beta = torch.linalg.solve(A.T @ A + l2 * torch.eye(F, dtype=A.dtype, device=A.device), A.T @ c)
    else:
        beta = A.T @ torch.linalg.solve(A @ A.T + l2 * torch.eye(M, dtype=A.dtype, device=A.device), c)
    return beta.to(out_dtype)
