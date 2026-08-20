# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file derive from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, used under the MIT License.
# The full MIT notice is reproduced in the NOTICE file at the repository root.

"""Transolver for unstructured point clouds, instrumented with the paper's ablations.

This is the reference model the kernel accelerates and the ablation study varies.
It is the unstructured path only: the reference implementation also supports
structured 1D/2D/3D meshes, but every experiment in the paper runs on point
clouds and the fused kernel supports only this path.

Defaults reproduce the original Transolver exactly. Each ablation flag changes
exactly one thing and at most one may be set at a time:

    no_token_attention        the softmax attention among slice tokens becomes a
                              per-token linear map -- tokens stop interacting
    untie_slice_weights       deslice gets its own projection and temperature
    share_slice_across_layers slice weights computed once, reused by all layers
    slice_once                slice once -> deep transformer on G tokens ->
                              deslice once (Perceiver limit; no point stream)
    mlp_only                  the attention sublayer is removed entirely

``use_fused_slice`` is orthogonal to all of them: an implementation switch, not
an ablation. It routes slice/deslice through the Triton kernels, leaves the
outputs unchanged, and produces interchangeable checkpoints.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from ..layers.basic import MLP
from ..layers.physics_attention import Physics_Attention_Irregular_Mesh


class TransolverBlock(nn.Module):
    """One encoder block: physics-attention residual, then a pointwise MLP residual."""

    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, untied_deslice=False,
                 no_token_attention=False, mlp_only=False, dim_head=None,
                 slice_dim_head=None, use_fused_slice=False):
        super().__init__()
        self.last_layer = last_layer
        self.mlp_only = mlp_only
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if not self.mlp_only:
            dh = dim_head if dim_head is not None else hidden_dim // num_heads
            self.Attn = Physics_Attention_Irregular_Mesh(
                hidden_dim, heads=num_heads, dim_head=dh, dropout=dropout,
                slice_num=slice_num, untied_deslice=untied_deslice,
                no_token_attention=no_token_attention, slice_dim_head=slice_dim_head,
                use_fused_slice=use_fused_slice)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, slice_weights_in=None):
        # Returns (fx, slice_weights). slice_weights is None under mlp_only, which
        # has no slice/deslice at all, and under the fused path, which never
        # materializes them.
        slice_weights = None
        if not self.mlp_only:
            attn_out, slice_weights = self.Attn(self.ln_1(fx),
                                                slice_weights_in=slice_weights_in)
            fx = attn_out + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx)), slice_weights
        return fx, slice_weights


class TokenTransformerBlock(nn.Module):
    """Pre-LN transformer block over slice tokens; used only by ``slice_once``."""

    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)

    def forward(self, tokens):
        h = self.ln_1(tokens)
        tokens = self.attn(h, h, h, need_weights=False)[0] + tokens
        tokens = self.mlp(self.ln_2(tokens)) + tokens
        return tokens


class Transolver(nn.Module):
    """Transolver on unstructured point clouds.

    Args:
        space_dim: coordinate dimension (2 or 3).
        fun_dim: number of per-point input features besides the coordinates.
        out_dim: number of predicted channels per point.
        n_hidden, n_heads, n_layers, slice_num, mlp_ratio, dropout, act:
            the usual backbone hyperparameters. ``slice_num`` is $G$, per head.
        dim_head: per-head width of the full-resolution path
            (default ``n_hidden // n_heads``). This is what drives the heavy
            ``[B, N, heads*dim_head]`` activations.
        slice_dim_head: width of the slice-token attention, which acts on $G$
            tokens only and is therefore nearly free in memory. Default: equal
            to ``dim_head``, i.e. the original model.

    Forward:
        ``model(x, fx)`` with ``x`` the coordinates ``[B, N, space_dim]`` and
        ``fx`` the per-point features ``[B, N, fun_dim]`` (or ``None``).
        Returns ``[B, N, out_dim]``, predicted at the input points.
    """

    def __init__(self, space_dim=3, fun_dim=1, out_dim=1, n_hidden=256, n_heads=8,
                 n_layers=8, slice_num=32, mlp_ratio=2, dropout=0.0, act="gelu",
                 dim_head: Optional[int] = None, slice_dim_head: Optional[int] = None,
                 untie_slice_weights=False, share_slice_across_layers=False,
                 slice_once=False, no_token_attention=False, mlp_only=False,
                 use_fused_slice=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                              n_layers=0, res=False, act=act)

        self.untied_deslice = untie_slice_weights
        self.share_slice_across_layers = share_slice_across_layers
        self.slice_once = slice_once
        self.no_token_attention = no_token_attention
        self.mlp_only = mlp_only
        n_flags = sum([untie_slice_weights, share_slice_across_layers, slice_once,
                       no_token_attention, mlp_only])
        if n_flags > 1:
            raise ValueError("At most one ablation flag may be enabled at a time.")

        self.use_fused_slice = use_fused_slice
        if use_fused_slice and (share_slice_across_layers or slice_once):
            raise ValueError("use_fused_slice is incompatible with "
                             "share_slice_across_layers and slice_once: both need "
                             "the slice weights the kernel deliberately never "
                             "materializes.")

        if slice_once:
            # Perceiver limit: one slice, all depth in token space, one deslice.
            self.slice_proj = nn.Linear(n_hidden, slice_num)
            self.slice_temperature = nn.Parameter(torch.tensor(0.5))
            self.token_blocks = nn.ModuleList([
                TokenTransformerBlock(num_heads=n_heads, hidden_dim=n_hidden,
                                      dropout=dropout, act=act, mlp_ratio=mlp_ratio)
                for _ in range(n_layers)])
            self.out_norm = nn.LayerNorm(n_hidden)
            self.out_head = nn.Linear(n_hidden, out_dim)
        else:
            self.blocks = nn.ModuleList([
                TransolverBlock(num_heads=n_heads, hidden_dim=n_hidden, dropout=dropout,
                                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                                slice_num=slice_num, last_layer=(i == n_layers - 1),
                                untied_deslice=untie_slice_weights,
                                no_token_attention=no_token_attention,
                                mlp_only=mlp_only, dim_head=dim_head,
                                slice_dim_head=slice_dim_head,
                                use_fused_slice=use_fused_slice)
                for i in range(n_layers)])
            if use_fused_slice and any(
                    getattr(getattr(b, "Attn", None), "fused_slice_fallback", None)
                    for b in self.blocks):
                # Some layer refused the dims and fell back to eager (it warned).
                # Report the EFFECTIVE state, so the flag is never silently inert
                # on a built model.
                self.use_fused_slice = False

        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _slice_once_forward(self, fx):
        slice_weights = torch.softmax(self.slice_proj(fx) / self.slice_temperature, dim=-1)
        slice_norm = slice_weights.sum(1)
        tokens = torch.einsum("bnc,bng->bgc", fx, slice_weights)
        tokens = tokens / (slice_norm + 1e-5).unsqueeze(-1)
        for block in self.token_blocks:
            tokens = block(tokens)
        out = torch.einsum("bgc,bng->bnc", tokens, slice_weights)
        out = out + fx  # point-level skip around the token stack
        return self.out_head(self.out_norm(out))

    def forward(self, x, fx=None):
        fx = self.preprocess(torch.cat((x, fx), dim=-1) if fx is not None else x)
        fx = fx + self.placeholder[None, None, :]

        if self.slice_once:
            return self._slice_once_forward(fx)

        shared_weights = None
        for block in self.blocks:
            fx, slice_weights = block(fx, slice_weights_in=shared_weights)
            if self.share_slice_across_layers and shared_weights is None:
                shared_weights = slice_weights
        return fx


__all__ = ["Transolver", "TransolverBlock", "TokenTransformerBlock"]
