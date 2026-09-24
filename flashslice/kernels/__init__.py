# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels for the slice/deslice bottleneck.

``fused_slice`` and ``fused_deslice`` are registered ``torch.library`` custom
ops, so they compose with autograd and stay a single dynamo graph.

Two kernel families sit behind them. The single-tile kernels in ``slice_ops``
hold the whole head width D and slice count G in one tile and serve both as
powers of two in [16, 128]; they are the tuned, measured path. The G-blocked
kernels in ``blocked`` serve every other shape — any G, D up to 256 — by
saving a per-point log-sum-exp and recomputing the slice weights one G-block
at a time. Routing is by shape; ``set_kernel_mode`` (or the
``FLASHSLICE_KERNEL_MODE`` environment variable) forces one path, and
``single_tile_dims`` says which path a shape takes by default.

``unsupported_dims(dim_head, slice_num)`` returns a reason string when
neither family can serve a shape and ``None`` when one can; the layer uses it
to fall back to the eager path loudly instead of failing.

``set_dot_mode`` selects the dot precision: ``"ieee"`` (default, exact fp32),
``"tf32"``, ``"bf16v"``, ``"bf16"`` or ``"tf32x3"``. It is an explicit opt-in
and is not tied to torch's ``allow_tf32``. ``set_stats_mode`` chooses how the
blocked kernels form the per-point softmax statistics when none are handed
in: ``"online"`` (default, one pass; a deslice forms its output in the same
pass) or ``"two-pass"`` (the original max-then-sum form).

Tile configurations are tuned per GPU class: ``tile_table()`` reports the
class in use (``"hopper"`` for GPUs with Hopper's shared memory per block,
``"ada"`` for smaller ones such as the RTX 4090) and ``set_tile_table``
(or ``FLASHSLICE_TILE_TABLE``) forces one.

The slice weight may be shared, per head or per sample and head — (G, D),
(H, G, D), (B, H, G, D) — and the logits width D may differ from the value
width of fx_mid and the tokens; both are documented on ``fused_slice``.

The implementation lives in ``slice_ops`` rather than ``fused_slice`` on
purpose: a module named after the function it exports shadows that function on
``from ... import fused_slice``, which is exactly the kind of ambiguity a
released package should not have.
"""

from .slice_ops import (
    fused_slice,
    fused_deslice,
    unsupported_dims,
    single_tile_dims,
    set_dot_mode,
    set_kernel_mode,
    set_tile_table,
    tile_table,
)
from . import blocked  # noqa: F401  — registers the flashslice::*_blk ops
from .blocked import set_block_g, set_stats_mode, stats_mode

__all__ = ["fused_slice", "fused_deslice", "unsupported_dims", "single_tile_dims",
           "set_dot_mode", "set_kernel_mode", "set_block_g", "set_stats_mode",
           "stats_mode", "set_tile_table", "tile_table"]
