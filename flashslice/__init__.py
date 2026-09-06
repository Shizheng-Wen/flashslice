"""FlashSlice placeholder: the eager reference of the slice/deslice operators.

The fused Triton kernels, the layer and the benchmarks are released with the
paper *Does Transolver Need a Transformer?*. See ``flashslice.reference``.
"""

__version__ = "0.0.1"

from .reference import deslice_eager, slice_eager  # noqa: F401

__all__ = ["slice_eager", "deslice_eager", "__version__"]
