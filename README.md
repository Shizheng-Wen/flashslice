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

**Documentation:** <https://shizheng-wen.github.io/flashslice/> — the design
(why slice/deslice is the cost, how the kernels stream it, the numerics gate),
shapes and routing, the API, the measured numbers, and how to run the tests
and sweeps. Tag `v0.1.0` is the tree the paper's numbers come from; `main`
carries the extensions listed in the [changelog](docs/changelog.md).

## Install

```bash
pip install -e .
```

Requires PyTorch and Triton. Developed and measured against torch 2.5 / Triton
3.0 on NVIDIA GH200 (Hopper); every number below is from that GPU.

Other GPUs are untested. Nothing in the kernels is Hopper-specific — the
`tf32` and `bf16` dot modes need Ampere-class tensor cores or newer, the
default `ieee` mode needs none — but the tile tables were swept on Hopper's
227 KB of shared memory per block, and entries with `num_stages=3` or
`BLOCK_N=256` can exceed the 100 KB of an Ada part such as the RTX 4090.
That fails at compile time with a Triton resource error, not silently; the
fix is a sweep on the target GPU (`bench/bench_kernels.py` for both kernel
families, then `bench/pick_tiles.py` for the blocked one) and new table
entries. Memory use does not depend on the GPU, so the sizes in the tables
below transfer as they are.

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

The kernels are usable on their own. `x_mid` `(B, N, H, D)` is what the
membership is computed from, `fx_mid` `(B, N, H, DV)` what is pooled, `W` the
slot projection, `b` a per-slot bias (or `None`), `tau` `(H,)` the temperature:

```python
from flashslice.kernels import fused_slice, fused_deslice, unsupported_dims

# slice first (Transolver's order); the deslice over the same membership
# takes the slice's statistics and skips its own pass
z_num, s, stats = fused_slice(x_mid, fx_mid, W, b, tau, return_stats=True)
tokens = mix(z_num / (s + 1e-5)[..., None])                  # (B, H, G, DV)
out = fused_deslice(x_mid, W, b, tau, tokens, stats=stats)   # (B, N, H, DV)

# deslice first (a persistent point stream); the deslice forms the
# statistics in its own online pass and the slice that follows takes them
out, stats = fused_deslice(x_mid, W, b, tau, tokens, return_stats=True)
z_num, s = fused_slice(x_mid, fx_mid, W, b, tau, stats=stats)
```

`W` may be `(G, D)`, `(H, G, D)` or `(B, H, G, D)` — shared, per head, or per
sample and head, e.g. `W = k(tokens)` — and its gradient comes back in that
shape; `b` likewise `None`, `(G,)`, `(H, G)` or `(B, H, G)`. `DV` may differ
from `D` (a membership carrying positional channels next to narrower values).

### Precision

Inputs are fp32, or bf16 under `torch.autocast`; those two are what the
parity gate covers (fp16 takes the same 16-bit path but is untested).
Parameters stay fp32, and accumulation is fp32 in every mode. What varies
is the precision of the dots, chosen with `set_dot_mode(...)` or the
`FLASHSLICE_DOT_MODE` environment variable:

| mode | dots | error against fp64 | use it for |
| --- | --- | --- | --- |
| `ieee` (default) | all fp32, on the FMA units | eager fp32's own | the paper's shapes; anything that must match eager |
| `tf32` | value dots on tensor cores, logits fp32 | ~5e-4 on outputs | fp32 inputs at large `G` |
| `bf16v` | value dots bf16, logits fp32 (16-bit inputs) | between the two | — |
| `bf16` | all dots bf16 (16-bit inputs) | eager bf16 autocast's own | bf16 training at large `G` |
| `tf32x3` | every dot as three tf32 products on tensor cores | 3–7× eager fp32's | fp32 inputs at large `G` when tf32's noise is too much |

The logits dot stays fp32 below the `bf16` level because the softmax
Jacobian amplifies noise in the slice weights. At the single-tile shapes the
default is also the fast choice; at large `G` the kernels are bound by dot
throughput and a tensor-core mode is what makes them faster than eager (see
the blocked table below).

`tf32x3` is the split-precision trick — `a·b ≈ a_hi·b_hi + a_hi·b_lo +
a_lo·b_hi` with tf32 parts — and Triton's implementation lands two to three
bits short of fp32 rather than on it: measured on outputs and every
gradient it is 3–7× eager's fp32 error, against tf32's ~1000×. It costs
three tensor-core products per dot plus the splitting, so it does not pay at
`G=32` (1.27× eager where `ieee` is 1.72×) and does at `G=256` with fp32
inputs, where it is the fastest mode (1.92× at `N=1M`).

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

Two kernel families sit behind `use_fused_slice`, and the layer picks by
shape:

- **Single-tile kernels** hold the whole head width `D` and slice count `G`
  in one tile, so the softmax over `G` never leaves registers. They serve `D`
  and `G` that are powers of two in `[16, 128]`, and they are the kernels
  every number above was measured with. Tile configurations are tuned per
  `G`: using the `G=32` table at `G=128` costs up to 40–80× through register
  spilling, so the tables are keyed by `G` rather than shared.
- **G-blocked kernels** serve every other shape: any `G` — the last block is
  masked, so `G=8`, `G=48` and `G=512` all run — any `D` up to 256, padded
  to a power of two, a value width `DV` apart from `D`, and weights per head
  or per sample. They take the approach of FlashAttention's *backward*
  rather than its forward. A small pass saves the per-point softmax
  statistics of the slice logits — row max and sum of exponentials, two
  floats per point and head, 2/`G` of `w` — in one online pass, and every
  kernel then recomputes `w = exp(logit − m) / l` one `G`-block at a time. A
  deslice with no statistics in hand forms its output and the statistics in
  that same pass (FlashAttention's forward) and hands them to a tied slice.
  Slice is the deslice's transpose — accumulators per token, summed over
  points — so the saved-statistics form is the one that serves both. Tokens,
  `dW` and `db` come from programs that own a `G`-block and stream over `N`;
  `out`, `dxm` and `dfx` from programs that own an `N`-block and stream over
  `G`. Still no atomics, still bitwise deterministic, held to the same parity
  gate. The price is the statistics pass (none in the deslice-first order)
  and one more logits recompute in the backward, which is why routing
  prefers the single-tile kernels wherever they apply.
  `set_kernel_mode("blocked")` (or `FLASHSLICE_KERNEL_MODE=blocked`) forces
  them, for timing and parity runs; `set_block_g` overrides the block size;
  `set_stats_mode("two-pass")` restores the original max-then-sum statistics.

Only `D > 256` is outside both. There the layer **falls back to the eager
path and says so** — it logs a warning, sets `use_fused_slice = False` on the
built model, and records the reason in `fused_slice_fallback`. A flag that is
silently inert trains a baseline replica and looks like a result; this one
cannot.

### The blocked kernels, measured

One layer (`Physics_Attention_Irregular_Mesh`, `H=8`, `D=32`), training step
unless marked, eager against fused on one GH200, medians of ten. The dot
mode is the `set_dot_mode` setting: `ieee` is the default, `bf16` needs
16-bit inputs.

| inputs / dots | *G* | *N* | eager | ours | speedup | eager mem | ours mem | saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 / bf16 | 256 | 262k | 47.5 ms | 11.9 ms | **3.98×** | 13.7 GB | 2.0 GB | 85% |
| bf16 / bf16 | 256 | 1M | 293.7 ms | 45.5 ms | **6.45×** | 54.5 GB | 8.1 GB | 85% |
| bf16 / bf16 | 512 | 262k | 90.6 ms | 20.0 ms | **4.53×** | 26.7 GB | 2.0 GB | 92% |
| bf16 / bf16 | 512 | 1M | OOM | 80.3 ms | — | — | 8.1 GB | — |
| bf16 / bf16 | 1024 | 262k | 180.2 ms | 37.6 ms | **4.79×** | 52.7 GB | 2.1 GB | 96% |
| bf16 / bf16, inference | 256 | 1M | 39.4 ms | 13.3 ms | **2.96×** | 17.0 GB | 2.1 GB | 88% |
| bf16 / bf16, inference | 256 | 4.2M | 157.3 ms | 53.2 ms | **2.95×** | 68.0 GB | 8.3 GB | 88% |
| bf16 / bf16 | 48 | 262k | 13.0 ms | 6.1 ms | **2.12×** | 3.1 GB | 2.0 GB | 35% |
| fp32 / tf32x3 | 256 | 262k | 45.4 ms | 37.4 ms | 1.21× | 14.8 GB | 2.0 GB | 86% |
| fp32 / tf32x3 | 256 | 1M | 283.6 ms | 147.7 ms | **1.92×** | 59.0 GB | 8.1 GB | 86% |
| fp32 / tf32 | 256 | 262k | 45.1 ms | 40.5 ms | 1.11× | 14.8 GB | 2.0 GB | 86% |
| fp32 / ieee | 256 | 262k | 45.1 ms | 53.2 ms | 0.85× | 14.8 GB | 2.0 GB | 86% |
| fp32 / ieee | 256 | 1M | 284.2 ms | 209.6 ms | 1.36× | 59.0 GB | 8.1 GB | 86% |
| bf16 / tf32 | 256 | 262k | 47.5 ms | 40.0 ms | 1.19× | 13.7 GB | 2.0 GB | 85% |
| bf16 / ieee | 256 | 262k | 47.4 ms | 64.0 ms | 0.74× | 13.7 GB | 2.0 GB | 85% |
| bf16 / ieee | 256 | 1M | 294.3 ms | 253.4 ms | 1.16× | 54.5 GB | 8.1 GB | 85% |

Reading it:

- **Memory behaves as at small `G`**: the fused layer sits at 2 GB whether
  `G` is 48 or 1024, while eager grows linearly and runs out at
  `G=512, N=1M`.
- **Sizing.** Fused memory scales with `B·N` and with the layer count, not
  with `G`: about 8 GB per million points per layer for a bf16 training
  step and 2 GB per million for inference (`H=8`, `D=32`); the model-level
  table above (8 layers, `G=32`) is the same arithmetic. By it, a 24 GB card
  trains the 8-layer bf16 model at roughly 350k points and runs inference
  near 4M — an extrapolation, not a measurement.
- **Time is a dot-throughput question.** Slice/deslice cost `O(N·G·D)`
  multiply-adds, and at `G=256` a point does eight times the work it does at
  `G=32`. Triton's `ieee` fp32 dot is an FMA path at ~15 TFLOP/s on this
  GPU, so in the default mode the blocked kernels are compute-bound and
  land around eager's time; on tensor cores they pull ahead — `bf16` for
  16-bit inputs by 3–6×, `tf32x3` for fp32 inputs by up to 1.9×. The
  single-tile kernels at `G=32` are near the memory/compute crossover,
  which is why the tables above did not need this choice.
- **Wide heads gain nothing.** At `D=256` the weight tensor is one eighth of
  the point activations and the blocked layer is 0.79× eager at 4% less
  memory; the kernels run there, but the bottleneck the paper found is not
  there.
- **Forced onto the paper shape** (`G=32`, where routing picks the
  single-tile kernels), the blocked kernels are 1.12× eager in fp32 against
  the single-tile 1.72×: the statistics pass and the extra logits recompute
  in the backward, and nothing else.

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
pytest tests/                       # dim contract, routing, visible fallback, eager equivalence
python bench/parity_test.py         # the layer: fused vs eager, fp64-referenced, both families
python bench/parity_ops.py          # the ops: weight layouts, value widths, both tied orders
python bench/bench_layer.py --slices 256 --dtype bf16   # one layer, eager vs fused, any shape
python bench/bench_kernels.py --defaults-only            # per-kernel time, bandwidth, TFLOP/s
python bench/bench_kernels.py --family blocked --slices 256   # sweep the blocked kernels
python bench/pick_tiles.py bench/results/sweep_*.json    # sweep results -> tile table entry
python bench/launch_probe.py --slices 256 --dtype bf16 --dot bf16   # each kernel in its own process
```

`bench/bench_kernels.py` reproduces the systems tables: it times eager against
fused for the forward, the training step, and inference, and sweeps the tile
configurations of either kernel family at any shape (`--dim-head`,
`--dim-value`, `--weight-shape`). `bench/bench_layer.py` times one layer,
eager against fused, at any shape, including the ones only the blocked
kernels serve. `bench/parity_test.py` checks that the fused layer matches
eager to the precision class of its dot mode, against an fp64 reference; the
`blk-tiny-N17` and `blk-g512` cases are the sensitive ones.
`bench/parity_ops.py` holds the ops to the same gate on what the layer does
not exercise. `bench/launch_probe.py` names the kernel behind a sticky CUDA
error by launching each one in a fresh process. The caveats that cost us time
are in `bench/README.md` and in the [documentation](https://shizheng-wen.github.io/flashslice/benchmarks/).

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
