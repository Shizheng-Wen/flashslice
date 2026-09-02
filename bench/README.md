# Benchmarks

Reproduce the paper's systems numbers. Both scripts need a GPU; they were run on
one NVIDIA GH200 (95 GB usable HBM) under torch 2.5 / Triton 3.0.

## Correctness first

```bash
python parity_test.py
```

Compares the fused path against eager on outputs and on every parameter
gradient, referenced to an fp64 computation, for each dot mode (`ieee`, `tf32`,
`bf16`). The gate is a relative-error class, not an absolute epsilon: fp32
fused must land at or below eager's own error against fp64. It also covers each
value of `G` separately, since the tile tables differ per `G`.

## Timing and memory

```bash
python bench_layer.py --slices 256 --dtype bf16              # one layer, eager vs fused, any shape
python bench_kernels.py --help
python bench_kernels.py --defaults-only                      # the shipped tile tables, per kernel
python bench_kernels.py --slices 64 --warps 4                # sweep one setting
python bench_kernels.py --family blocked --slices 256        # sweep the G-blocked kernels
```

`bench_layer.py` is the user's view: one `Physics_Attention_Irregular_Mesh`,
forward or training step, eager against fused in the same process, time and
peak memory. `bench_kernels.py` is the per-kernel view for tuning; its
`--defaults-only` mode prints time, bandwidth and TFLOP/s of every kernel at
the shipped tiles, which is how to tell a memory-bound kernel from a
compute-bound one.

Two facts from those numbers that shape any tuning here:

- **The ieee dot path is FMA, not tensor cores.** Even at `G=32` the
  single-tile kernels run at ~15 TFLOP/s and a quarter of the copy bandwidth,
  so they are compute-limited, and the blocked kernels at `G=256` do eight
  times the dot work per point. `set_dot_mode("bf16")` (16-bit inputs),
  `"tf32"` or `"tf32x3"` moves the dots to tensor cores; at `G=256` that is
  the difference between slower than eager and several times faster.
  `tf32x3` was the attempt to get fp32 accuracy on tensor cores: it lands
  at 3–7x eager's fp32 error (not parity), is slower than `ieee` at `G=32`
  and the fastest fp32 mode at `G=256`.
- **Backward kernels with tensor-core dots must stay on 4 warps.** Triton 3.0
  aborts the process (an assert, not an exception) on an mma -> mma layout
  conversion when such a kernel is compiled for 8 warps; both tile lookups
  clamp it and the sweep skips those configurations.
- **The blocked kernels are more tile-sensitive than the single-tile ones.**
  With borrowed tiles two of them ran 9-11x slower than their single-tile
  twins at the same shape; a two-dimensional load mask (needed only when D
  is padded) costs the vectorized loads and with them the layout the FMA dot
  path wants. Both are handled — masks are compile-time flags and the family
  has its own tile table — but a new shape deserves a `--family blocked`
  sweep before its numbers are trusted.

Notes that cost us time and may cost you some:

- **Run one benchmark job at a time.** Two processes over the same GPU measure
  each other, not the kernel. A `G=128` probe once came out slower than `G=256`
  for exactly this reason.
- **Medians of at least ten iterations**, and compare eager and fused *inside
  one job*: cross-job comparisons drift with clocks and placement.
- `--verbose` prints each configuration before launching it. Triton 3.0 can
  abort the process rather than raise on some tile/warp combinations, and the
  printed line is what tells you which one did it.
