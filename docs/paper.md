# What the paper found

The kernel exists because of a claim about what the layer is doing. The claim
is that the load-bearing structure of Transolver is the *coupling* —
alternating full-resolution pointwise MLPs with the slice/deslice
bottleneck — and not the self-attention among slice tokens.

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

- **`no_token_attention` costs nothing** — −2.6% to +1.8% on six benchmarks,
  +3.8% and +7.9% on the other two. Replacing the content-dependent attention
  core with a constant learned matrix is free, and on several benchmarks it is
  the best variant.
- **The coupling is not optional.** `mlp_only` — the same model with the whole
  attention sublayer removed — loses 1.04× to 5.6×, and widening it to 112% of
  the baseline's parameters still leaves it 1.6–1.9× short. That is an
  expressivity limit, not a capacity one.
- **Nor is the point stream.** `slice_once` keeps the attention and *more*
  parameters than the baseline but collapses the points into token space; it
  is the worst variant on every benchmark (2.2×–6.8×).

Which is why the kernel targets slice/deslice: it is the part that turned out
to matter, and it was the part that dominated memory. Every variant is one
flag on `flashslice.Transolver` ([API reference](api.md#the-layer-and-the-model)).

## Citation

```bibtex
@inproceedings{flashslice,
  title  = {Does Transolver Need a Transformer?},
  note   = {under review},
  year   = {2027}
}
```

## Attribution

The physics-attention layer, the MLP and the block/backbone structure derive
from [Transolver](https://github.com/thuml/Transolver) (Copyright (c) 2024
THUML @ Tsinghua University), MIT License; `NOTICE` reproduces the notice in
full. The Triton kernels, the ablation flags and the instrumentation are
original work, Apache-2.0. LinearNO (Hu et al., AAAI 2026) is not vendored;
the paper compares against the authors' own release at
<https://github.com/HiPRL/LinearNO>.
