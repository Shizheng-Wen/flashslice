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
python bench_kernels.py --help
python bench_kernels.py --defaults-only          # the shipped tile tables
python bench_kernels.py --slices 64 --warps 4    # sweep one setting
```

Notes that cost us time and may cost you some:

- **Run one benchmark job at a time.** Two processes over the same GPU measure
  each other, not the kernel. A `G=128` probe once came out slower than `G=256`
  for exactly this reason.
- **Medians of at least ten iterations**, and compare eager and fused *inside
  one job*: cross-job comparisons drift with clocks and placement.
- `--verbose` prints each configuration before launching it. Triton 3.0 can
  abort the process rather than raise on some tile/warp combinations, and the
  printed line is what tells you which one did it.
