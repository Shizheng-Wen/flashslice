# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""FlashSlice: fused slice/deslice kernels for physics-attention, and the
Transolver layer they accelerate.

Companion code for "Does Transolver Need a Transformer?".

    from flashslice import Transolver
    model = Transolver(space_dim=3, fun_dim=1, out_dim=4, use_fused_slice=True)

The kernels are importable on their own if you want to drop them into a
different implementation:

    from flashslice.kernels import fused_slice, fused_deslice, unsupported_dims
"""

from .layers.physics_attention import Physics_Attention_Irregular_Mesh
from .models.transolver import Transolver, TransolverBlock, TokenTransformerBlock

__all__ = [
    "Transolver",
    "TransolverBlock",
    "TokenTransformerBlock",
    "Physics_Attention_Irregular_Mesh",
]
__version__ = "0.1.0"
