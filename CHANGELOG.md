# Changelog

## Unreleased

- **Hopper tile tables per Triton series.** The Hopper tables were swept
  under Triton 3.0; under Triton 3.6 (torch 2.11) they put the default `ieee`
  kernels 2x to 9x behind their best, slower than eager. Triton 3.6 and newer
  now take a set of their own on Hopper (single-tile `G = 16-128` at the ieee,
  tf32-class and bf16 levels; blocked `(32, 32)`, `(64, 32)`, `(64, 64, 32)`),
  swept on a GH200; `tile_series()` reports the series, `set_tile_series` /
  `FLASHSLICE_TILE_SERIES` force one. The Triton 3.0 set and the Ada tables
  are untouched. Triton 3.1 to 3.5 on Hopper run on the 3.0 set with a
  warning. Numbers under both stacks are in the README.
- **`bench/replay_tiles.py`** replays the layer's own backward call at every
  tile of one blocked kernel. `bench_kernels.py` hands the blocked deslice
  backward fp32 tokens, the layer under bf16 autocast hands it bf16 ones, and
  Triton compiles the two apart: under Triton 3.6 the sweep's winner for that
  kernel was four times slower inside a training step than in the sweep.

## v0.3.0 — 2026-09-29

First full release on PyPI (0.0.1 was a placeholder), with the paper on
arXiv: [2609.32525](https://arxiv.org/abs/2609.32525).

- **Tile tables per GPU class.** `tile_table()` picks the Hopper tables
  (unchanged) on GPUs with Hopper's shared memory per block and new Ada
  tables, swept on an RTX 4090, on smaller ones; `set_tile_table` /
  `FLASHSLICE_TILE_TABLE` force one. The Ada set covers the single-tile
  family (ieee, bf16 and tf32-class dots, G = 16-128) and the blocked family
  ((32, 32), (64, 32) and (16, 256) keys). On the 4090 several Hopper tiles
  did not compile (shared memory) and others were up to 4x slower at the
  layer level.
- On Ada, `tf32x3` at G >= 128 routes to the blocked family: the single-tile
  deslice backward needs 128 KB of shared memory there at any tile size.
- The docs site is gone; the README is self-contained and is also the PyPI
  project description. `CITATION.cff` and the README carry the arXiv citation.

## v0.2.0 — 2026-09-06

Extensions that came from using the kernels as the coupling of a point-cloud
model with sample-specific anchors (see the README, *A coupling with sample-specific slots*).
The single-tile kernels' numerical path is untouched; the blocked family
changed and is re-gated. What comes next is in [ROADMAP.md](ROADMAP.md).

- **Weight layouts.** `weight` may be `(G, D)`, `(H, G, D)` or `(B, H, G, D)`;
  `bias` `None`, `(G,)`, `(H, G)` or `(B, H, G)`. Read through strides, zero
  where shared; `dW`/`db` come back in the parameter's shape and dtype.
- **Value width apart from the logits width.** `fx_mid`, `tokens` and `out`
  have their own width `DV` on the blocked family; unequal widths route
  there. `unsupported_dims` and `single_tile_dims` take it as an optional
  argument.
- **One-pass statistics.** The blocked statistics kernel forms `(m, l)` in
  one online pass; `set_stats_mode("two-pass")` keeps the original form for
  attribution.
- **Online deslice.** A deslice with no statistics forms `out` and `(m, l)`
  in one pass; `fused_deslice(..., return_stats=True)` returns them and
  `fused_slice(..., stats=)` takes them, so a tied coupling pays for the
  membership once per op in either order.
- **Correctly rounded reciprocals** (`div_rn`) for `1/tau` and `1/l` where
  the temperature gradient can see them; on the FMA paths the temperature
  rides on the once-loaded operand of the logits dot and the normalizers on
  `dl` and the accumulators, the slot-owning kernels keep dividing their
  tile (the one form whose transposed FMA dot stays fast).
- **Wider G-block on tensor-core paths.** `tiles(D, G, dot)` allows
  `GB * D_tile <= 4096` there (2048 on FMA paths); a swept
  `(G_block=64, D_tile=64, DV_tile=32)` entry serves the anchor-coupling shape.
- **Tile table** keys may carry the value width: `(G_block, D_tile, DV_tile)`.
- **Bench**: `bench_kernels.py --dim-value --weight-shape --stages`, the new
  kernels; `pick_tiles.py` emits three-element keys; `bench/parity_ops.py`.

## v0.1.0 — the paper's tree

The kernels as submitted with *Does Transolver Need a Transformer?*:
single-tile fused slice/deslice for `G`, `D` powers of two in `[16, 128]`,
the G-blocked kernels for every other shape, `ieee` / `tf32` / `bf16v` /
`bf16` / `tf32x3` dot modes, the instrumented Transolver layer with the
paper's ablation flags, the parity gate and the benches. Every number in the
README of that tag is from it, on one GH200 under torch 2.5 / Triton 3.0.
