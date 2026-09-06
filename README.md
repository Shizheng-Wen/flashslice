# FlashSlice

Fused Triton kernels for the slice/deslice bottleneck of physics-attention
(Transolver-style layers): the membership tensor that couples every mesh point
to every slice token is never written to memory, and the layer's memory stops
growing with the token count.

**This is a placeholder release.** The kernels, the instrumented Transolver
layer, the parity gate and the benchmarks are released together with the
paper *Does Transolver Need a Transformer?* (under review); until then the
repository is private and this package holds only the eager reference of the
two operators, so that the interface is on record:

```python
from flashslice.reference import slice_eager, deslice_eager
```

`slice_eager(x_mid, fx_mid, weight, bias, tau)` pools point features onto
slots through a softmax membership; `deslice_eager(x_mid, weight, bias, tau,
tokens)` broadcasts slot values back to the points through the same
membership. The fused kernels implement exactly these two maps and their
gradients.

Apache-2.0. Copyright 2026 Shizheng Wen.
