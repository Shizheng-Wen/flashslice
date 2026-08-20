"""Fused Triton kernels for the slice/deslice bottleneck.

``fused_slice`` and ``fused_deslice`` are registered ``torch.library`` custom
ops, so they compose with autograd and stay a single dynamo graph.

``unsupported_dims(dim_head, slice_num)`` returns a reason string when the
kernels cannot serve a shape and ``None`` when they can; the layer uses it to
fall back to the eager path loudly instead of failing.

``set_dot_mode`` selects the dot precision: ``"ieee"`` (default, exact fp32),
``"tf32"``, ``"bf16v"`` or ``"bf16"``. It is an explicit opt-in and is not tied
to torch's ``allow_tf32``.

The implementation lives in ``slice_ops`` rather than ``fused_slice`` on
purpose: a module named after the function it exports shadows that function on
``from ... import fused_slice``, which is exactly the kind of ambiguity a
released package should not have.
"""

from .slice_ops import (
    fused_slice,
    fused_deslice,
    unsupported_dims,
    set_dot_mode,
)

__all__ = ["fused_slice", "fused_deslice", "unsupported_dims", "set_dot_mode"]
