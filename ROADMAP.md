# Roadmap

What FlashSlice is for, in one line: keep the slice/deslice coupling of
physics-attention out of memory and at the speed of the GPU's dot units, for
any slot count and any weight layout, with a parity gate that holds the fused
path to eager's own distance from fp64.

Status of each item: **done** (released), **next** (the next release),
**later** (queued, design settled), **research** (open question, needs an
experiment before a design).

## v0.2.0 — done

The round that came from using the kernels as the anchor-keyed coupling of a
point-cloud model (measured on that model: one training step 985 → 387 ms at
265k points, 1024 anchors, of which the kernels are 0.50–0.55×).

- Slice weights shared, per head or per sample and head, with the gradient
  flowing back to whatever produced them.
- A value width apart from the logits width.
- One-pass online statistics; an online deslice that hands its statistics to
  a tied slice, so a tied coupling pays for the membership once per op in
  either order.
- Correctly rounded reciprocals where the temperature gradient can see the
  rounding; the FMA paths' arithmetic settled by measurement.
- Tile-table keys with the value width, per-kernel G-blocks, a 4096 block
  budget on tensor-core paths; the anchor-coupling shape swept.
- `bench/parity_ops.py`, `bench/launch_probe.py`, DV-aware sweeps; the docs
  site.

## Next — v0.3

Kernel throughput on the tensor-core paths. At the anchor-coupling shape the
kernels run at 11–13% of the GH200's bf16 peak; the slot-owning backward
kernels are now half of a coupling round.

- [ ] **Profile before tuning.** One `ncu` pass over the eight blocked
      kernels at the anchor shape: instruction mix, MUFU (exp) share, shared
      memory traffic from layout conversions, occupancy. Every tile decision
      so far was made from end-to-end timings.
- [ ] **Slot-owning backward kernels** (`slice_bwd_g`, `deslice_bwd_g`):
      the `P`-partition split over `N` and the per-program partial buffers
      were designed for `G ≤ 512`; at `G = 1024`–`4096` with `B·H·ngb` programs
      the partition count collapses to 8. Sweep `P`, the partial-buffer layout
      and the tiles at `N ∈ {66k, 265k, 1M}`.
- [ ] **A paper-shape sweep at `D = 64`** (`G = 256`, fp32 and bf16): the
      `(32, 64)` and `(64, 64)` keys still borrow the single-tile tables.
- [ ] **Triton ≥ 3.2 in a newer container.** The 4-warp cap on backward
      kernels with tensor-core dots, the single-stage tf32 dots and the
      mma → mma assert are Triton 3.0 limits; a newer Triton may lift all
      three and pipelines Hopper's wgmma. Re-sweep, re-gate, record.
- [ ] **Exponentials.** At `D + DV ≈ 90` the exp per (point, slot) is the
      second cost after the dots (FlashAttention-3's small-head-dimension
      problem). Fold `log2 e` and the temperature into the logits operand,
      use `exp2`, and measure whether the MUFU share moves.

## Later — design settled

- [ ] **Tied backward fusion.** In a deslice → MLP → slice coupling both ops
      share `x_mid`, `W` and the statistics. The slice's `dfx` pass must run
      before the MLP's backward, but its `dx_mid`, `dW` and `dτ` can be
      deferred and merged into the deslice's backward: one recompute of the
      logits and one `dx_mid` / `dW` dot for both ops. Backward passes over
      the membership per coupling go from 6 to 3. Needs a custom autograd
      node that stitches the two ops; the kernels are a merge of the two
      backward pairs.
- [ ] **Ragged batching.** Samples of different `N` (and `G`) in one launch
      through row splits, so that small meshes fill the GPU instead of
      running one sample per device behind a fixed per-step floor.
- [ ] **Point-sharded slice, documented.** `fused_slice` already returns the
      unnormalized partials `(z_num, s)` and the deslice is local given the
      full tokens; a sharded coupling is an all-reduce of `(z_num, s)` in the
      forward and of `dtokens` in the backward. Ship the autograd wrapper and
      an example rather than a kernel change.
- [ ] **Off-mesh deslice, documented.** The kernels take a query-side `x_mid`
      with its own `N` today; add the example and the test.
- [ ] **A 16-bit output path.** `out`, `dx_mid` and `dfx` follow `x_mid`'s
      dtype already; audit the remaining fp32 intermediates the layer keeps
      per point when training in bf16.

## Research — needs an experiment first

- [ ] **Block-sparse membership.** With points and slots sorted in space,
      the membership of an anchor-keyed coupling is concentrated: most
      (point-block, slot-block) pairs contribute nothing to the softmax. A
      per-pair skip with an exact online-softmax error bound would make the
      cost ∝ `N × active blocks` instead of `N × G`, recover kNN-like FLOPs
      while keeping the membership differentiable, and fall back to dense
      where the bound fails. The open questions are the bound (positional
      channels give one; content channels do not) and whether training keeps
      the membership sparse.
- [ ] **fp8 logits.** Hopper's fp8 tensor cores double the dot rate; the
      softmax Jacobian's sensitivity to logit noise (why the logits dot stays
      fp32 below the bf16 level today) is the question.
- [ ] **A hand-written kernel** (CUTLASS / ThunderKittens style, warp
      specialization, TMA) for the four core passes, if the Triton ceiling
      after v0.3 is still far from the dot units' peak.

## Housekeeping

- [x] PyPI release from a tag (`.github/workflows/release.yml`, trusted
      publishing; the trusted publisher on pypi.org is a one-time setup).
- [ ] A GPU-less CI job for the tests that do not need a GPU (dims contract,
      routing, eager fallback); the parity gate stays a job on a GPU.
- [ ] Per-release performance tables in the docs, with the job ids.
