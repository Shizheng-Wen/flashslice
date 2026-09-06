# Performance

All numbers from one NVIDIA GH200 (95 GB usable HBM) under torch 2.5 /
Triton 3.0, medians of ten, eager against fused *inside one job*. The
constant factor is not the interesting part; what changes is how the layer
scales. Reproduce with `bench/bench_layer.py` and `bench/bench_kernels.py`
([Tests and benchmarks](benchmarks.md)).

## The paper's model, single-tile kernels

Eager PyTorch against FlashSlice in the configuration every experiment in the
paper runs in: eager, no compilation, no activation checkpointing; 8 layers,
`G = 32`, `H = 8`, `D = 32`.

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

- **Memory is flat in the slice count.** From `G=16` to `G=128` the fused
  layer moves 15.42 → 15.43 GB while eager goes 17.92 → 36.89 GB (bf16). The
  tensor that is never written is the only per-layer term that grows with
  `G`, so raising it is free for us and linear for eager.
- **Depth reaches further.** At a 95 GB budget: 32 layers where eager fits 16
  (fp32), 48 where eager fits 32 (bf16). The two compound — at `G=128` eager
  stops at 16 layers in both precisions while we reach 32 (fp32) and 48
  (bf16).

## Large slice counts, blocked kernels

One layer (`H = 8`, `D = 32`), training step unless marked. The dot mode is
the `set_dot_mode` setting; `ieee` is the default, `bf16` needs 16-bit inputs.

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

- **Memory behaves as at small `G`**: 2 GB whether `G` is 48 or 1024, while
  eager grows linearly and runs out at `G=512, N=1M`. Fused memory scales with
  `B·N` and the layer count, not with `G`: about 8 GB per million points per
  layer for a bf16 training step, 2 GB per million for inference.
- **Time is a dot-throughput question.** Slice/deslice cost `O(N·G·D)`
  multiply-adds, and at `G=256` a point does eight times the work it does at
  `G=32`. Triton's `ieee` fp32 dot is an FMA path at ~15 TFLOP/s on this GPU,
  so in the default mode the blocked kernels are compute-bound and land around
  eager's time; on tensor cores they pull ahead — `bf16` by 3–6×, `tf32x3` for
  fp32 inputs by up to 1.9×.
- **Wide heads gain nothing.** At `D=256` the weight tensor is one eighth of
  the point activations and the blocked layer is 0.79× eager at 4% less
  memory; the bottleneck the paper found is not there.
- **Forced onto the paper shape** (`G=32`), the blocked kernels are 1.12×
  eager in fp32 against the single-tile 1.72×: the statistics pass and the
  extra logits recompute in the backward, and nothing else.

## A coupling with sample-specific slots

The extensions on `main` came from using the kernels as the coupling of a
point-cloud model whose slots are physical anchor points of each sample:
weight `(1, H, G, D)` computed from the anchors' token embeddings, `D = 56`
(32 content + 24 positional channels), `DV = 32`, `H = 8`, `N ≈ 265k`,
`G` from 256 to 16 384, bf16 dots in training. Measured facts from that use,
kernels at tag `v0.1.0` plus the weight/value-width extension:

- One coupling round (deslice + slice, forward + backward), `G = 256`:
  175 ms `ieee` / 31 ms `bf16`; `G = 2048`: 1365 / 209 ms; `G = 16384`:
  10.5 s / 1.5 s. Peak memory 2.2–3.0 GB, flat in `G`.
- In the model, the coupling was 80% of a training step at `G = 1024` and
  ran at roughly 6% of the GPU's bf16 peak: at this shape the blocked kernels
  had no tuned tile entry and borrowed one, and a tied coupling ran six
  logits passes per layer in the forward alone (two-pass statistics for each
  op, plus a checkpointed recompute).
- `bf16` dots matched a `tf32` control at every step of a 50k-step training
  run, at 3.5× less step time.

### After the 2026-09 round

One coupling round (deslice first, its statistics handed to the slice;
forward + backward), `N = 265k`, `H = 8`, `D = 56`, `DV = 32`, bf16 dots,
medians of five, one GH200:

| `G` | v0.1.0 kernels (+ weight/value-width extension) | `main` | |
| --- | --- | --- | --- |
| 256 | 30.8 ms | 14.6 ms | **2.1×** |
| 1024 | 107 ms | 48.3 ms | **2.2×** |
| 2048 | 208 ms | 93.1 ms | **2.2×** |

Per kernel at `G = 1024` (ms, bf16 dots), before → after:

| kernel | before | after | what changed |
| --- | --- | --- | --- |
| statistics | 9.3 (two passes) | 2.6 | one online pass; tiles |
| deslice, no statistics in hand | 9.3 + 5.9 | 2.2 | the online kernel forms both |
| slice forward | 17.5 | 6.5 | tiles, `G`-block 64 |
| slice backward, points | 18.3 | 8.1 | tiles, `G`-block 64 |
| slice backward, slots | 10.0 | 9.7 | tiles |
| deslice backward, points | 21.4 | 7.0 | tiles, `G`-block 64 |
| deslice backward, slots | 14.9 | 13–15 | tiles |

The slot-owning backward kernels are now the largest share of the round;
they stream over `N` in `P` partitions and their tiles were swept at one
`N`, so they are where the next round of tuning goes. The `ieee` path at
this shape gained 1.25× from the one-pass statistics; the paper's shapes
(`D = 32`, `G = 32` single-tile and `G = 256` blocked) are unchanged within
noise, layer-level, in every dot mode.
