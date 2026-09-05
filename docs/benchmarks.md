# Tests and benchmarks

Everything below needs a GPU except `pick_tiles.py` and the parts of the test
suite that exercise the eager path.

```bash
pytest tests/                        # dim contract, routing, visible fallback, eager equivalence
python bench/parity_test.py          # the layer: fused vs eager, fp64-referenced, both families
python bench/parity_ops.py           # the ops: weight layouts, value widths, both tied orders
python bench/parity_ops.py --orders  # every call order of the two ops survives a backward
python bench/parity_ops.py --time    # one tied coupling round at the anchor-coupling shape
python bench/bench_layer.py --slices 256 --dtype bf16    # one layer, eager vs fused, any shape
python bench/bench_kernels.py --defaults-only            # per-kernel time, bandwidth, TFLOP/s
python bench/bench_kernels.py --family blocked --slices 256     # sweep the blocked kernels
python bench/pick_tiles.py bench/results/sweep_*.json    # sweep results -> tile table entry
```

## Correctness first

`bench/parity_test.py` compares the fused layer against eager on outputs and
every parameter gradient, referenced to an fp64 computation, for each dot mode
and each `G` the tables are tuned for, on both kernel families
([the gate](design/numerics.md#the-gate)). The `blk-tiny-N17` and `blk-g512`
cases are the sensitive ones: at `N = 17` the temperature gradient's
cancellation has nothing to average over.

`bench/parity_ops.py` holds the ops to the same gate on what the layer does
not exercise: per-head and per-sample weights (the latter computed from a
token stream, so the gradient must flow through `dW`), a value width apart
from the logits width, the deslice-first order with statistics handed to the
slice and the slice-first order with statistics handed to the deslice, a
masked non-power-of-two `G`, `G = 1`, and the bf16 modes. `--stats-mode
two-pass` runs the same cases on the original statistics kernel, to attribute
a difference. `--orders` builds every call order of the two ops in a fresh
graph and runs one backward each; `--time` times one tied coupling round with
and without the statistics handed on.

## Timing and memory

`bench/bench_layer.py` is the user's view: one layer, forward or training
step, eager against fused in the same process, time and peak memory, at any
shape including the ones only the blocked kernels serve.

`bench/bench_kernels.py` is the per-kernel view for tuning. Its
`--defaults-only` mode prints time, bandwidth and TFLOP/s of every kernel at
the shipped tiles, which is how to tell a memory-bound kernel from a
compute-bound one. Without it the script sweeps `BLOCK_N × num_warps ×
num_stages × dot mode` per kernel and writes a JSON. Flags: `--family
single|blocked`, `--slices G`, `--dim-head D`, `--dim-value DV`,
`--weight-shape gd|hgd|bhgd`, `--block-g`, `--dtypes`, `--dots`, `--ns`,
`--bns`, `--warps`, `--stages`, `--verbose`.

## Tile sweeps

A new blocked shape deserves its own tile entry
([why](design/shapes.md#tile-tables)):

```bash
# sweep at the shape, one dtype and dot mode at a time is enough
python bench/bench_kernels.py --family blocked --slices 1024 --dim-head 56 --dim-value 32 \
    --weight-shape bhgd --dtypes bf16 --dots bf16 --ns 262144 --verbose \
    --out bench/results/sweep_blk_d56dv32_g1024_bf16.json
# print the _CFG_BLK entry and paste it into flashslice/kernels/blocked.py
python bench/pick_tiles.py bench/results/sweep_blk_d56dv32_g1024_bf16.json
```

The entry is keyed by `(G_block, D_tile)` or, when the value width differs,
`(G_block, D_tile, DV_tile)`. When several `N` were swept the winner is the
configuration with the lowest mean slowdown against the per-`N` best.

## Notes that cost us time

- **Run one benchmark job at a time.** Two processes over the same GPU
  measure each other, not the kernel. A `G=128` probe once came out slower
  than `G=256` for exactly this reason.
- **Medians of at least ten iterations**, and compare eager and fused *inside
  one job*: cross-job comparisons drift with clocks and placement.
- **The `ieee` dot path is FMA, not tensor cores.** Even at `G=32` the
  single-tile kernels run at ~15 TFLOP/s and a quarter of the copy bandwidth,
  so they are compute-limited; the blocked kernels at `G=256` do eight times
  the dot work per point. `set_dot_mode("bf16")` (16-bit inputs), `"tf32"` or
  `"tf32x3"` moves the dots to tensor cores.
- **Backward kernels with tensor-core dots must stay on 4 warps** under
  Triton 3.0 (an assert, not an exception, aborts the process on 8); the
  lookups clamp it and the sweep skips those configurations. tf32 dots run
  single-stage for the same kind of reason.
- **`--verbose` prints each configuration before launching it.** Triton 3.0
  can abort the process rather than raise on some tile/warp combinations, and
  the printed line is what tells you which one did it.
- **Do not share a `randn(...).requires_grad_()` divided by a constant as a
  leaf** in a timing loop: the division makes it a non-leaf whose backward
  node frees its saved divisor after the first step, and the second step
  raises "backward through the graph a second time".
