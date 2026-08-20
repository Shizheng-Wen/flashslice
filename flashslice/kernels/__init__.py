"""Fused Triton kernels for the slice/deslice bottleneck.

``fused_slice`` and ``fused_deslice`` are registered ``torch.library`` custom
ops, so they compose with autograd and stay a single dynamo graph.
``unsupported_dims(dim_head, slice_num)`` returns a reason string when the
kernels cannot serve a shape, and ``None`` when they can -- the layer uses it to
fall back to the eager path loudly instead of failing.
"""

from .fused_slice import fused_slice, fused_deslice, unsupported_dims

__all__ = ["fused_slice", "fused_deslice", "unsupported_dims"]
