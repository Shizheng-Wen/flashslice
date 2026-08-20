# FlashSlice

Fused Triton kernels for the slice/deslice bottleneck of physics-attention, and
the Transolver layer they accelerate.

Companion code for **“Does Transolver Need a Transformer?”**. The paper's claim
is that Transolver's accuracy comes from alternating full-resolution pointwise
MLPs with a learned low-rank pooling/unpooling bottleneck — and *not* from the
self-attention among physics tokens. This repository ships the two artifacts a
reader would want to run: the kernel that makes the bottleneck cheap, and the
layer instrumented with every ablation the paper reports.

It is deliberately not a training framework. There is no trainer, no data
pipeline, no configs — just the model and the kernels, so they drop into
whatever you already use.

## Install

```bash
pip install -e .
```

Requires PyTorch and Triton. Developed and measured against torch 2.5 / Triton
3.0 on NVIDIA GH200 (Hopper). The kernels use Hopper tensor-core paths; other
architectures are untested.

## Quick start

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

The kernels are usable on their own:

```python
from flashslice.kernels import fused_slice, fused_deslice, unsupported_dims
```

## What the kernel does

The eager layer materializes the slice-weight tensor `w` of shape
`(B, H, N, G)` — for `N` in the millions that is the dominant activation, and
it is stored for the backward pass. FlashSlice makes the layer two streaming
passes over `N` around a small token state, **never writes `w`**, and recomputes
it in the backward pass. That is FlashAttention's trade, applied to
slice/deslice rather than to softmax attention.

It is an implementation switch, not a model change: outputs match the eager path
and checkpoints interchange in both directions.

Measured against eager PyTorch on one GH200, at the paper's configuration:

|                      | speedup            | memory      |
| -------------------- | ------------------ | ----------- |
| training step, fp32  | 1.35×              | −14–25%     |
| training step, bf16  | 1.53–1.70×         | −14–25%     |
| inference            | 1.52–1.68×         | −20–26%     |

And it changes how the layer scales, which matters more than the constant:

- **Memory is flat in `G`.** From `G=16` to `G=128` the fused layer moves
  15.42 → 15.43 GB; eager goes 17.92 → 36.89 GB. The tensor that is never
  written is the only per-layer term that grows with the slice count.
- **Depth reaches further.** At a 95 GB budget, 32 layers where eager fits 16
  (fp32), 48 where eager fits 32 (bf16).

### Supported shapes

The kernels hold the whole head width `D` and the slice count `G` in one tile,
so both must be powers of two in `[16, 128]`. Outside that range the layer
**falls back to the eager path and says so** — it logs a warning, sets
`use_fused_slice = False` on the built model, and records the reason in
`fused_slice_fallback`. A flag that is silently inert trains a baseline replica
and looks like a result; this one cannot.

Tile configurations are tuned per `G`. Using the `G=32` table at `G=128` costs
up to 40–80× through register spilling, so the tables are keyed by `G` rather
than shared.

## Ablations

Every variant in the paper is one flag on `Transolver`, and at most one may be
set at a time:

| flag | what changes |
| --- | --- |
| `no_token_attention` | attention among slice tokens becomes a per-token linear map — tokens stop interacting |
| `untie_slice_weights` | deslice gets its own projection and temperature |
| `share_slice_across_layers` | slice weights computed once at layer 1, reused by all layers |
| `slice_once` | slice once → deep transformer on `G` tokens → deslice once (Perceiver limit) |
| `mlp_only` | the attention sublayer is removed entirely (pointwise lower bound) |

`use_fused_slice` is orthogonal to all of them, except that it is refused
together with `share_slice_across_layers` and `slice_once`: both consume the
slice weights the kernel deliberately never materializes, so they raise at
construction rather than quietly producing a different model.

## Tests and benchmarks

```bash
pytest tests/                       # dim contract, visible fallback, eager equivalence
python bench/parity_test.py         # fused vs eager, fp64-referenced
python bench/bench_kernels.py --help
```

`bench/bench_kernels.py` reproduces the systems tables: it times eager against
fused for the forward, the training step, and inference, and sweeps the tile
configurations. `bench/parity_test.py` checks that the fused path matches eager
to the precision class of its dot mode, against an fp64 reference.

## Attribution

The Transolver layer and backbone derive from the reference implementation of
Transolver (Wu et al.); the ablation flags, the fused kernels, and the
instrumentation are ours. If you compare against LinearNO (Hu et al., AAAI
2026), please use their own release at
<https://github.com/HiPRL/LinearNO> rather than a reimplementation.

> **TODO before making this public: add a `LICENSE`.** The code derives from
> Transolver's release, so check its terms and keep them compatible.

## Citation

```bibtex
@inproceedings{flashslice,
  title  = {Does Transolver Need a Transformer?},
  note   = {under review},
  year   = {2027}
}
```
