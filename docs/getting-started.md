# Getting started

## Install

```bash
git clone https://github.com/Shizheng-Wen/flashslice
cd flashslice
pip install -e .
```

Requires PyTorch (2.4 or newer) and Triton (3.0 or newer); `einops` is pulled
in for the layer. Nothing in the kernels is Hopper-specific, but the tile
tables were swept on a GH200 and other GPUs are untested: see
[Shapes and routing](design/shapes.md#other-gpus).

## The layer

```python
import torch
from flashslice import Transolver

model = Transolver(
    space_dim=3, fun_dim=1, out_dim=4,      # coords, per-point inputs, outputs
    n_hidden=256, n_heads=8, n_layers=8,
    slice_num=32,                            # G, per head
    use_fused_slice=True,                    # route slice/deslice through Triton
).cuda()

x  = torch.randn(1, 200_000, 3, device="cuda")   # coordinates
fx = torch.randn(1, 200_000, 1, device="cuda")   # per-point features
y  = model(x, fx)                                # [1, 200000, 4]
```

`use_fused_slice` changes nothing but the implementation. A shape the kernels
cannot serve makes the layer fall back to the eager path *and say so*: a
warning, `use_fused_slice` reading `False` on the built model, and the reason
in `fused_slice_fallback`. The ablation flags of the paper are documented in
the [API reference](api.md#the-layer-and-the-model).

## The ops

The kernels are usable without the layer. Shapes: `x_mid` `(B, N, H, D)` is
what the membership is computed from, `fx_mid` `(B, N, H, DV)` what is pooled,
`weight` the slot projection, `bias` per slot, `tau` `(H,)` the temperature.

```python
from flashslice.kernels import fused_slice, fused_deslice

# slice first (Transolver's order): pool points onto G tokens, hand the
# statistics to the deslice that shares the membership
z_num, s, stats = fused_slice(x_mid, fx_mid, W, b, tau, return_stats=True)
tokens = mix(z_num / (s + 1e-5)[..., None])         # (B, H, G, DV)
out = fused_deslice(x_mid, W, b, tau, tokens, stats=stats)   # (B, N, H, DV)

# deslice first (a persistent point stream): the deslice forms the
# statistics in its own pass and the slice that follows takes them
out, stats = fused_deslice(x_mid, W, b, tau, tokens, return_stats=True)
z_num, s = fused_slice(x_mid, fx_mid, W, b, tau, stats=stats)
```

`weight` may be `(G, D)`, `(H, G, D)` or `(B, H, G, D)`; `bias` `None`,
`(G,)`, `(H, G)` or `(B, H, G)`. A per-sample weight computed from a token
stream gets its gradient back through the op. `DV` may differ from `D`.

## Precision

Inputs are fp32, or bf16 under `torch.autocast`. Parameters stay fp32 and
accumulation is fp32 in every mode; what varies is the precision of the dots,
chosen with `set_dot_mode(...)` or `FLASHSLICE_DOT_MODE`:

| mode | dots | error against fp64 | use it for |
| --- | --- | --- | --- |
| `ieee` (default) | all fp32, on the FMA units | eager fp32's own | the paper's shapes; anything that must match eager |
| `tf32` | value dots on tensor cores, logits fp32 | ~5e-4 on outputs | fp32 inputs at large `G` |
| `bf16v` | value dots bf16, logits fp32 (16-bit inputs) | between the two | — |
| `bf16` | all dots bf16 (16-bit inputs) | eager bf16 autocast's own | bf16 training at large `G` |
| `tf32x3` | every dot as three tf32 products on tensor cores | 3–7× eager fp32's | fp32 inputs at large `G` when tf32's noise is too much |

The logits dot stays fp32 below the `bf16` level because the softmax Jacobian
amplifies noise in the slice weights. At the paper's shapes the default is also
the fast choice; at large `G` the kernels are bound by dot throughput and a
tensor-core mode is what makes them faster than eager
([Performance](performance.md)).

!!! note "TF32 in containers"
    Some container images set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which
    makes every fp32 cuBLAS matmul run in TF32 whatever the Python flags say.
    Under such an image "fp32 eager" is TF32 everywhere except in the kernels'
    `ieee` path, and a fused-vs-eager difference of a few 1e-4 is the eager
    side. Compare against an fp64 reference, or unset the variable.

## Environment variables

| variable | values | effect |
| --- | --- | --- |
| `FLASHSLICE_DOT_MODE` | `ieee` `tf32` `bf16v` `bf16` `tf32x3` | dot precision, as `set_dot_mode` |
| `FLASHSLICE_KERNEL_MODE` | `auto` `single-tile` `blocked` | force a kernel family, as `set_kernel_mode` |
| `FLASHSLICE_STATS_MODE` | `online` `two-pass` | how the blocked kernels form the softmax statistics, as `set_stats_mode` |
