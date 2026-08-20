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

Measured against eager PyTorch on one GH200, in the configuration every
experiment in the paper runs in — eager, no compilation, no activation
checkpointing:

| | *N* | eager | ours | speedup | eager mem | ours mem | saved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **training step** (fwd + bwd) | | | | | | | |
| fp32 | 262k | 171 ms | 126 ms | **1.35×** | 28.0 GB | 24.0 GB | 14% |
| bf16 | 262k | 143 ms | 93.1 ms | **1.53×** | 20.4 GB | 15.4 GB | 24% |
| bf16 | 1M | 585 ms | 345 ms | **1.70×** | 81.3 GB | 61.3 GB | 25% |
| **inference** (fwd only) | | | | | | | |
| fp32 | 1M | 229 ms | 138 ms | **1.66×** | 10.1 GB | 8.1 GB | 20% |
| fp32 | 8.4M | 1944 ms | 1160 ms | **1.68×** | 80.5 GB | 64.5 GB | 20% |
| bf16 | 1M | 174 ms | 113 ms | **1.53×** | 7.6 GB | 5.6 GB | 26% |
| bf16 | 8.4M | 1420 ms | 932 ms | **1.52×** | 60.5 GB | 44.5 GB | 26% |
| bf16 | 12.6M | OOM | 1411 ms | — | — | 66.7 GB | — |

The constant factor is not the interesting part. What changes is how the layer
*scales*:

![Memory and time against slice count and depth](assets/F4_systems.png)

- **Memory is flat in the slice count.** From `G=16` to `G=128` the fused layer
  moves 15.42 → 15.43 GB while eager goes 17.92 → 36.89 GB (bf16). The tensor
  that is never written is the only per-layer term that grows with `G`, so
  raising it is free for us and linear for eager.
- **Depth reaches further.** At a 95 GB budget: 32 layers where eager fits 16
  (fp32), 48 where eager fits 32 (bf16). The two compound — at `G=128` eager
  stops at 16 layers in both precisions while we reach 32 (fp32) and 48 (bf16).

Reproduce with `bench/bench_kernels.py`; the caveats that cost us time are in
`bench/README.md`.

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

## What the paper found

The kernel exists because of a claim about what the layer is doing. The claim is
that the load-bearing structure is the *coupling* — alternating full-resolution
pointwise MLPs with the slice/deslice bottleneck — and not the self-attention
among slice tokens.

![Ablation ratios across eight benchmarks](assets/F1_money.png)

Best validation relative L¹ across eight benchmarks in fluid dynamics and
industrial aerodynamics, up to 1.4×10⁸ mesh points. Three seeds for the first
two columns, one seed otherwise; **bold** is the best per row.

| Dataset | Baseline | NoTokenAttn | MlpOnly | Untied | FrozenSlice | SliceOnce |
| --- | --- | --- | --- | --- | --- | --- |
| Taylor–Green | 0.0756 ±.0001 | **0.0745** ±.0001 | 0.0786 | 0.0755 | 0.0777 | 0.188 |
| SHIFT-Wing surface | 0.0580 ±.0005 | 0.0579 ±.0005 | 0.139 | **0.0576** | 0.0577 | 0.155 |
| DrivAerNet++ surface | 0.193 ±.005 | **0.188** ±.002 | 0.294 | 0.190 | 0.192 | 0.435 |
| DrivAerNet++ volume | 0.1625 ±.0003 | **0.159** ±.001 | 0.315 | 0.162 | 0.162 | 0.386 |
| SHIFT-SUV surface | 0.15654 ±.00074 | 0.15738 ±.00003 | 0.252 | **0.15575** | 0.15923 | 0.449 |
| SHIFT-SUV volume | 0.06010 ±.00056 | 0.06238 ±.00086 | 0.197 | **0.05773** | 0.06834 | 0.408 |
| DrivAerML surface | 0.089 ±.001 | 0.096 ±.000 | 0.501 | **0.086** | 0.104 | 0.579 |
| DrivAerML volume | 0.110 ±.005 | 0.112 ±.004 | 0.497 | **0.108** | 0.123 | 0.745 |

Reading it:

- **`no_token_attention` costs nothing** — −2.6% to +1.8% on six benchmarks,
  +3.8% and +7.9% on the other two. Replacing the content-dependent attention
  core with a constant learned matrix is free, and on several benchmarks it is
  the best variant.
- **The coupling is not optional.** `mlp_only` — the same model with the whole
  attention sublayer removed — loses 1.04× to 5.6×, and widening it to 112% of
  the baseline's parameters still leaves it 1.6–1.9× short. That is an
  expressivity limit, not a capacity one.
- **Nor is the point stream.** `slice_once` keeps the attention and *more*
  parameters than the baseline but collapses the points into token space; it is
  the worst variant on every benchmark (2.2×–6.8×).

Which is why the kernel targets slice/deslice: it is the part that turned out to
matter, and it was the part that dominated memory.

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

## License and attribution

Apache License 2.0 — see [`LICENSE`](LICENSE).

The physics-attention layer, the MLP and the block/backbone structure derive
from [Transolver](https://github.com/thuml/Transolver) (Copyright (c) 2024
THUML @ Tsinghua University), used under the MIT License. Those files carry a
header saying so and [`NOTICE`](NOTICE) reproduces the MIT notice in full, as
that license requires. The Triton kernels, the ablation flags and the
instrumentation are original work.

LinearNO (Hu et al., AAAI 2026) is not vendored here. The paper compares against
it; use the authors' own release at <https://github.com/HiPRL/LinearNO>.

## Citation

```bibtex
@inproceedings{flashslice,
  title  = {Does Transolver Need a Transformer?},
  note   = {under review},
  year   = {2027}
}
```
