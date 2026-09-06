"""Eager reference of the two operators the FlashSlice kernels fuse.

Shapes: ``x_mid`` (B, N, H, D) is what the membership is computed from,
``fx_mid`` (B, N, H, DV) what is pooled, ``weight`` (G, D) | (H, G, D) |
(B, H, G, D) the slot projection, ``bias`` None | (G,) | (H, G) | (B, H, G),
``tau`` (H,) a per-head temperature, ``tokens`` (B, H, G, DV) the slot values
to broadcast. The membership ``w`` (B, H, N, G) is materialized here; the
fused kernels never write it.
"""

import torch


def membership(x_mid, weight, bias, tau):
    """softmax over the slots of (x_mid . weight + bias) / tau, (B, H, N, G)."""
    B, N, H, D = x_mid.shape
    w4 = weight.reshape((1,) * (4 - weight.dim()) + tuple(weight.shape)).expand(B, H, -1, -1)
    logits = torch.einsum("bnhd,bhgd->bhng", x_mid, w4.to(x_mid.dtype))
    if bias is not None:
        b3 = bias.reshape((1,) * (3 - bias.dim()) + tuple(bias.shape))
        logits = logits + b3.to(logits.dtype)[:, :, None, :]
    return torch.softmax(logits / tau.to(logits.dtype).view(1, H, 1, 1), dim=-1)


def slice_eager(x_mid, fx_mid, weight, bias, tau):
    """Pool point features onto slots. Returns the unnormalized ``z_num``
    (B, H, G, DV) and the membership mass ``s`` (B, H, G); the slot values are
    ``z_num / (s + eps)[..., None]``."""
    w = membership(x_mid, weight, bias, tau)
    z_num = torch.einsum("bhng,bnhd->bhgd", w, fx_mid.to(w.dtype))
    return z_num, w.sum(dim=2)


def deslice_eager(x_mid, weight, bias, tau, tokens):
    """Broadcast slot values back to the points through the same membership:
    ``out`` (B, N, H, DV)."""
    w = membership(x_mid, weight, bias, tau)
    return torch.einsum("bhng,bhgd->bnhd", w, tokens.to(w.dtype))
