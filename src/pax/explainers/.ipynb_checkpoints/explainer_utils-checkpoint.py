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