# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""G-blocked slice/deslice kernels: any slice count, head width up to 256.

The single-tile kernels in ``slice_ops`` hold the whole slice axis G in one
tile, so the softmax over G is register-local and needs no bookkeeping — and
that is also what caps them at G <= 128. This module lifts the cap the way
FlashAttention-2's *backward* does, not the way its forward does:

* A small pass computes the per-point softmax statistics of the slice
  logits — the row max ``m[n]`` and the sum of exponentials ``l[n]`` —
  streaming over G in blocks of ``GB``, max first and then the sum against
  that max. That is two floats per point and head, 2/G of the tensor the
  eager path stores, and it is kept for the backward.
* Every other kernel recomputes ``w[n, g] = exp(logit[n, g] - m[n]) / l[n]``
  one G-block at a time. Nothing is normalized across blocks in-register, so
  there is no rescaling and no running statistic to carry.
* Quantities indexed by g (tokens, dW, db, dz') come from programs that own
  a G-block and stream over N; quantities indexed by n (out, dxm, dfx) from
  programs that own an N-block and stream over G. The softmax Jacobian in
  the backward needs the per-point term ``delta[n] = sum_g w[n,g] dw[n,g]``;
  the N-owned kernels compute it in fp32 and hand it to the G-owned ones.

Online softmax in the FlashAttention-forward sense — rescale a running
accumulator whenever the row max moves — only works when the accumulator is
indexed by the softmax's own row. Deslice is that shape; slice is its
transpose (accumulators per token, summed over points), so the saved-
statistics form is the one that serves both. It is the same reason
FlashAttention's backward saves LSE instead of streaming it.

One gradient is far more sensitive to rounding than the rest and shaped
this module: the temperature's, sum_n sum_g dl * (-logit / tau), where
dl = w (dw - delta) sums to zero over g on every row in exact arithmetic.
Anything that breaks that identity coherently across a row — a row of w
that does not sum to one, or a delta that is not the sum of the very same
rounded products w_g dw_g that dl is built from — becomes a bias scaled by
the logits, while every other gradient shrugs it off. Hence: the pair
(m, l) rather than lse = m + log l (exp(logit - lse) rounds the argument of
the dominant weights); l summed against the final max in a second pass
rather than rescaled online; and delta computed from w * dw block by block
in the N-owned kernels, not from the algebraically equal fx . dfx + w @ ds.
The last was decisive — 3x eager's error at N=17, G=256 before, parity
after — and the other two are cheap insurance of the same kind.

Cost against the single-tile path at a shape both can serve: one extra read
of x_mid for the statistics pass, one more recompute of the logits in each
backward (the N-owned kernels make two passes over G), and the G-owned
kernels re-read their inputs once per G-block. The block index is the
fastest grid axis, so consecutive programs stream the same rows and the
re-reads mostly resolve in L2. That is why routing prefers the single-tile
kernels wherever they apply, and why those kernels are left untouched.

D is held whole, padded to a power of two and masked, up to 256 — the choice
FlashAttention makes for the head dimension. G can be anything >= 1: the
last block is masked. Both paddings are compile-time flags (``PAD_D``,
``PAD_G``) and the masks exist only when set: a two-dimensional load mask
that Triton cannot prove constant along d costs the vectorized loads, and
with them the layout the FMA dot path needs — a 5-10x slowdown of the ieee
kernels when this was first measured. Tiles come from this family's own
sweep table (``_CFG_BLK``), and where it has no entry from the single-tile
tables keyed by the block size GB, since a block of GB slices has the
register profile of a whole tile of G = GB. No atomics anywhere:
per-program partial sums, reduced with a deterministic ``.sum()``.
"""

import torch
import triton
import triton.language as tl

from .slice_ops import _cfg, _dot, _dot_w, _n_programs, _stages, _strides

_BLOCK_G = None  # None = choose from D; set_block_g overrides


def set_block_g(gb):
    """Force the G-block size (a power of two >= 16), or None for automatic:
    the largest of 16..64 that keeps GB * D_tile <= 2048."""
    global _BLOCK_G
    if gb is not None and (gb < 16 or (gb & (gb - 1)) != 0):
        raise ValueError("block size must be a power of two >= 16, got %r" % gb)
    _BLOCK_G = gb


def _pow2_at_least_16(v):
    return max(16, triton.next_power_of_2(v))


def tiles(d, g):
    """(D_tile, G_block) for the blocked kernels at head width d, slice count g."""
    dt = _pow2_at_least_16(d)
    gb = _BLOCK_G or max(16, min(64, 2048 // dt))
    return dt, min(gb, _pow2_at_least_16(g))


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #

@triton.jit
def _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D: tl.constexpr,
             PAD_G: tl.constexpr, PAD_D: tl.constexpr):
    """One G-block of the slice projection: (GB, DT) weight rows and bias.
    Rows beyond G and columns beyond D read as zero."""
    ptr = W + offs_g[:, None] * D + offs_d[None, :]
    if PAD_G or PAD_D:
        w_mat = tl.load(ptr, mask=gmask[:, None] & dmask[None, :], other=0.0)
    else:
        w_mat = tl.load(ptr)
    bias = _load_vec(BS, offs_g, G)
    return w_mat.to(tl.float32), bias


@triton.jit
def _rows_ptr(X, b, h, offs_n64, offs_d, sb, sn, sh, sd):
    return X + b * sb + h * sh + offs_n64[:, None] * sn + offs_d[None, :] * sd


@triton.jit
def _load_rows(X, b, h, offs_n64, offs_d, nmask, dmask, sb, sn, sh, sd,
               PAD_D: tl.constexpr):
    """(BN, DT) rows of a (B, N, H, D) tensor, read through its strides. The
    mask is per row unless D is padded, so the loads stay vectorized."""
    ptr = _rows_ptr(X, b, h, offs_n64, offs_d, sb, sn, sh, sd)
    if PAD_D:
        x = tl.load(ptr, mask=nmask[:, None] & dmask[None, :], other=0.0)
    else:
        x = tl.load(ptr, mask=nmask[:, None], other=0.0)
    return x.to(tl.float32)


@triton.jit
def _store_rows(X, val, b, h, offs_n64, offs_d, nmask, dmask, sb, sn, sh, sd,
                PAD_D: tl.constexpr):
    """Store (BN, DT) rows into a (B, N, H, D) tensor, in its dtype."""
    ptr = _rows_ptr(X, b, h, offs_n64, offs_d, sb, sn, sh, sd)
    if PAD_D:
        tl.store(ptr, val.to(X.dtype.element_ty),
                 mask=nmask[:, None] & dmask[None, :])
    else:
        tl.store(ptr, val.to(X.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _load_tok(T, bh64, offs_g, offs_d, gmask, dmask, G, D: tl.constexpr,
              PAD_G: tl.constexpr, PAD_D: tl.constexpr):
    """One G-block of a (B, H, G, D) token tensor."""
    ptr = T + bh64 * G * D + offs_g[:, None] * D + offs_d[None, :]
    if PAD_G or PAD_D:
        t = tl.load(ptr, mask=gmask[:, None] & dmask[None, :], other=0.0)
    else:
        t = tl.load(ptr)
    return t.to(tl.float32)


@triton.jit
def _load_vec(V, offs, limit):
    """A vector indexed by offs, read with the offsets clamped to limit - 1.

    Never a masked load: Triton 3.0 mis-assigns layouts for a masked
    one-dimensional load with a fill value whose result is then broadcast
    against a two-dimensional tile ("arith.select op expected condition type
    to have the same shape as the result type", make_ttgir). Out-of-range
    lanes read a valid neighbour instead; every consumer masks them out."""
    return tl.load(V + tl.minimum(offs, limit - 1)).to(tl.float32)


@triton.jit
def _store_part(P, val, idx, offs_g, offs_d, gmask, dmask, G, D: tl.constexpr,
                PAD_G: tl.constexpr, PAD_D: tl.constexpr):
    """One program's (GB, DT) partial into rows [g0, g0+GB) of a (.., G, D)
    buffer."""
    ptr = P + idx * G * D + offs_g[:, None] * D + offs_d[None, :]
    if PAD_G or PAD_D:
        tl.store(ptr, val, mask=gmask[:, None] & dmask[None, :])
    else:
        tl.store(ptr, val)


@triton.jit
def _logits(xm, w_mat, bias, tau, DOT: tl.constexpr):
    """(BN, GB) slice logits: bias before the temperature division, as eager."""
    return (_dot_w(xm, tl.trans(w_mat), DOT) + bias[None, :]) / tau


@triton.jit
def _load_stats(STATS, bh64, N, offs_n64):
    """Row max m and exponential sum l of the softmax, (BN,) each. Rows past
    N read row N - 1 (see _load_vec): finite weights that every consumer
    masks away."""
    base = STATS + bh64 * 2 * N
    m = _load_vec(base, offs_n64, N)
    l = _load_vec(base + N, offs_n64, N)
    return m, l


@triton.jit
def _weights(lg, m, l):
    """Softmax rows from logits and their statistics: exp(lg - m) / l."""
    return tl.exp(lg - m[:, None]) / l[:, None]


@triton.jit
def _stats_kernel(
    XM, W, BS, TAU, STATS,
    N, G, H,
    sxb, sxn, sxh, sxd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """m[n] = max_g logit[n, g], l[n] = sum_g exp(logit[n, g] - m[n]);
    stored as STATS[b, h, 0, n] and STATS[b, h, 1, n].

    Two passes over G rather than one online pass: the max first, then the
    sum against the final max. An online sum rescales its running value by
    exp(m_old - m_new) at every block, and each rescale costs a rounding
    that the consumers' own recomputation of exp(logit - m) does not share,
    so their rows summed to 1 with three times the error of the single-tile
    kernels — and the temperature gradient, which is sum dl * logit with
    sum_g dl = -delta * (row sum - 1), showed exactly that factor (3.7e-6
    against eager's 1.3e-6 at N=17, G=256). Summed against the final max,
    l is the same 256-term sum the consumers implicitly form, to the
    precision of one summation. The extra logits pass is on the cheapest
    kernel of the seven."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    tau = tl.load(TAU + h).to(tl.float32)

    m = tl.full((BN,), float("-inf"), tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, tau, DOT)
        if PAD_G:
            lg = tl.where(gmask[None, :], lg, float("-inf"))
        m = tl.maximum(m, tl.max(lg, axis=1))
    l = tl.zeros((BN,), tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        e = tl.exp(_logits(xm, w_mat, bias, tau, DOT) - m[:, None])
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        l += tl.sum(e, axis=1)
    base = STATS + bh.to(tl.int64) * 2 * N
    tl.store(base + offs_n64, m, mask=nmask)
    tl.store(base + N + offs_n64, l, mask=nmask)


@triton.jit
def _slice_fwd_g_kernel(
    XM, FX, W, BS, TAU, STATS, PART_Z, PART_S,
    N, G, P, H,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one G-block, streams over N: z_num[g] = sum_n w[n,g] fx[n],
    s[g] = sum_n w[n,g]. Per-program partials, reduced on the host."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    tau = tl.load(TAU + h).to(tl.float32)

    acc_z = tl.zeros((GB, DT), dtype=tl.float32)
    acc_s = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        fx = _load_rows(FX, b, h, offs_n64, offs_d, nmask, dmask,
                        sfb, sfn, sfh, sfd, PAD_D)
        acc_z += _dot(tl.trans(w), fx, DOT)
        acc_s += tl.sum(w, axis=0)

    idx = (bh * P + pid).to(tl.int64)
    _store_part(PART_Z, acc_z, idx, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    if PAD_G:
        tl.store(PART_S + idx * G + offs_g, acc_s, mask=gmask)
    else:
        tl.store(PART_S + idx * G + offs_g, acc_s)


@triton.jit
def _deslice_fwd_n_kernel(
    XM, W, BS, TAU, TOK, STATS, OUT,
    N, G, H,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one N-block, streams over G: out[n] = sum_g w[n,g] z'[g]."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    m, l = _load_stats(STATS, bh64, N, offs_n64)
    tau = tl.load(TAU + h).to(tl.float32)

    acc = tl.zeros((BN, DT), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(gmask[None, :], w, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        acc += _dot(w, tok, DOT)
    _store_rows(OUT, acc, b, h, offs_n64, offs_d, nmask, dmask, sob, son, soh, sod,
                PAD_D)


@triton.jit
def _slice_bwd_n_kernel(
    XM, FX, W, BS, TAU, STATS, DZN, DS,
    DXM, DFX, DELTA, PDT,
    N, G, H,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one N-block. Pass 1 over G: dfx = w @ dz_num and the Jacobian
    term delta = sum_g w dw. Pass 2: dxm, and the temperature gradient.

    delta is formed from the same rounded products w_g * dw_g that pass 2
    and the G-owned kernel put into dl = w (dw - delta), so that sum_g dl
    vanishes to the precision of one summation, as it does in the
    single-tile kernels. The algebraically equal delta = fx . dfx + w @ ds
    (one dot cheaper) rounds differently, and when sum_g w dw cancels
    internally its deviation becomes a bias shared by every dl of the row;
    the temperature gradient, sum dl * logit, showed it as 3x eager's error
    at N=17, G=256 while every other gradient was unaffected.

    dtau lives here and not in the G-owned kernel because it is
    sum_n sum_g dl * (-logit / tau) with sum_g dl = 0 on every row: the row
    sums cancel almost completely, and they must be formed row by row, over
    all of G, before anything is added across rows — the order the
    single-tile kernels use. Summing each G-block's share over N first and
    cancelling across blocks at the end put 2-3x eager's error on this
    gradient (N=17, G=256 and G=512), while every other gradient matched."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    fx = _load_rows(FX, b, h, offs_n64, offs_d, nmask, dmask, sfb, sfn, sfh, sfd,
                    PAD_D)
    m, l = _load_stats(STATS, bh64, N, offs_n64)
    tau = tl.load(TAU + h).to(tl.float32)

    acc_dfx = tl.zeros((BN, DT), dtype=tl.float32)
    delta = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(gmask[None, :], w, 0.0)
        dzn = _load_tok(DZN, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        ds = _load_vec(DS + bh64 * G, offs_g, G)
        acc_dfx += _dot(w, dzn, DOT)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        delta += tl.sum(w * dw, axis=1)
    _store_rows(DFX, acc_dfx, b, h, offs_n64, offs_d, nmask, dmask,
                sfb, sfn, sfh, sfd, PAD_D)
    tl.store(DELTA + bh64 * N + offs_n64, delta, mask=nmask)

    acc_dxm = tl.zeros((BN, DT), dtype=tl.float32)
    row_dt = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, tau, DOT)
        w = _weights(lg, m, l)
        if PAD_G:
            w = tl.where(gmask[None, :], w, 0.0)
        dzn = _load_tok(DZN, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        ds = _load_vec(DS + bh64 * G, offs_g, G)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        dl = w * (dw - delta[:, None])
        acc_dxm += _dot(dl / tau, w_mat, DOT)
        row_dt += tl.sum(dl * (-lg / tau), axis=1)
    _store_rows(DXM, acc_dxm, b, h, offs_n64, offs_d, nmask, dmask,
                sxb, sxn, sxh, sxd, PAD_D)
    tl.store(PDT + bh64 * tl.num_programs(0) + pid,
             tl.sum(tl.where(nmask, row_dt, 0.0), axis=0))


@triton.jit
def _slice_bwd_g_kernel(
    XM, FX, W, BS, TAU, STATS, DELTA, DZN, DS,
    PDW, PDB,
    N, G, P, H,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one G-block, streams over N: partial dW and db."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    tau = tl.load(TAU + h).to(tl.float32)
    dzn = _load_tok(DZN, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    ds = _load_vec(DS + bh64 * G, offs_g, G)

    acc_dw = tl.zeros((GB, DT), dtype=tl.float32)
    acc_db = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        fx = _load_rows(FX, b, h, offs_n64, offs_d, nmask, dmask,
                        sfb, sfn, sfh, sfd, PAD_D)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        delta = _load_vec(DELTA + bh64 * N, offs_n64, N)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        dlr = w * (dw - delta[:, None]) / tau
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)

    idx = (bh * P + pid).to(tl.int64)
    _store_part(PDW, acc_dw, idx, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    if PAD_G:
        tl.store(PDB + idx * G + offs_g, acc_db, mask=gmask)
    else:
        tl.store(PDB + idx * G + offs_g, acc_db)


@triton.jit
def _deslice_bwd_n_kernel(
    XM, W, BS, TAU, TOK, STATS, DOUT,
    DXM, DELTA, PDT,
    N, G, H,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one N-block. Pass 1 over G: delta = sum_g w (dout . z'[g]) in
    fp32 (recomputed rather than read back from a rounded `out`). Pass 2:
    dxm and the temperature gradient (row by row over all of G, for the
    reason given on _slice_bwd_n_kernel)."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    dout = _load_rows(DOUT, b, h, offs_n64, offs_d, nmask, dmask, sob, son, soh,
                      sod, PAD_D)
    m, l = _load_stats(STATS, bh64, N, offs_n64)
    tau = tl.load(TAU + h).to(tl.float32)

    delta = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(gmask[None, :], w, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        delta += tl.sum(w * _dot(dout, tl.trans(tok), DOT), axis=1)
    tl.store(DELTA + bh64 * N + offs_n64, delta, mask=nmask)

    acc_dxm = tl.zeros((BN, DT), dtype=tl.float32)
    row_dt = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, tau, DOT)
        w = _weights(lg, m, l)
        if PAD_G:
            w = tl.where(gmask[None, :], w, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        dw = _dot(dout, tl.trans(tok), DOT)
        dl = w * (dw - delta[:, None])
        acc_dxm += _dot(dl / tau, w_mat, DOT)
        row_dt += tl.sum(dl * (-lg / tau), axis=1)
    _store_rows(DXM, acc_dxm, b, h, offs_n64, offs_d, nmask, dmask,
                sxb, sxn, sxh, sxd, PAD_D)
    tl.store(PDT + bh64 * tl.num_programs(0) + pid,
             tl.sum(tl.where(nmask, row_dt, 0.0), axis=0))


@triton.jit
def _deslice_bwd_g_kernel(
    XM, W, BS, TAU, TOK, STATS, DELTA, DOUT,
    PDTOK, PDW, PDB,
    N, G, P, H,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
):
    """Owns one G-block, streams over N: partial dz', dW and db."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    tau = tl.load(TAU + h).to(tl.float32)
    tok = _load_tok(TOK, bh64, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)

    acc_dtok = tl.zeros((GB, DT), dtype=tl.float32)
    acc_dw = tl.zeros((GB, DT), dtype=tl.float32)
    acc_db = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        dout = _load_rows(DOUT, b, h, offs_n64, offs_d, nmask, dmask,
                          sob, son, soh, sod, PAD_D)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        delta = _load_vec(DELTA + bh64 * N, offs_n64, N)
        w = _weights(_logits(xm, w_mat, bias, tau, DOT), m, l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        acc_dtok += _dot(tl.trans(w), dout, DOT)
        dw = _dot(dout, tl.trans(tok), DOT)
        dlr = w * (dw - delta[:, None]) / tau
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)

    idx = (bh * P + pid).to(tl.int64)
    _store_part(PDTOK, acc_dtok, idx, offs_g, offs_d, gmask, dmask, G, D,
                PAD_G, PAD_D)
    _store_part(PDW, acc_dw, idx, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    if PAD_G:
        tl.store(PDB + idx * G + offs_g, acc_db, mask=gmask)
    else:
        tl.store(PDB + idx * G + offs_g, acc_db)


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #

# (BLOCK_N, num_warps, num_stages) per (G_block, D_tile) -> kernel ->
# (input-is-16bit, dot level), from bench_kernels.py --family blocked sweeps
# on one GH200 at N=262k (see the job numbers next to each table). The
# blocked kernels are more tile-sensitive than the single-tile ones: with
# borrowed tiles the ieee slice_fwd_g ran 9x and deslice_bwd_g 11x slower
# than their single-tile twins at G=32, and the winners below bring the
# whole family to about twice the single-tile kernel time at that shape —
# the statistics pass and the recomputes, nothing else.
# Entries are (input-is-16bit, dot level); a missing level falls back as
# _cfg_blk describes, and a missing (G_block, D_tile) borrows the single-tile
# table for G = G_block through _FAMILY.
#   (32, 32): job 3262609 (G=32; fp32 and bf16 sweeps).
#   (64, 32): jobs 3262609 (G=256, fp32) and 3263142 (G=256, bf16).
_CFG_BLK = {
    (32, 32): {
        "stats": {
            (False, 0): (256, 4, 1),
            (False, 1): (256, 4, 1),
            (True, 0): (256, 4, 1),
            (True, 1): (256, 4, 1),
            (True, 3): (256, 4, 2),
        },
        "slice_fwd_g": {
            (False, 0): (128, 4, 3),
            (False, 1): (128, 4, 1),
            (True, 0): (128, 4, 2),
            (True, 1): (128, 4, 1),
            (True, 3): (256, 4, 3),
        },
        "deslice_fwd_n": {
            (False, 0): (256, 4, 3),
            (False, 1): (128, 4, 1),
            (True, 0): (256, 4, 3),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 1),
        },
        "slice_bwd_n": {
            (False, 0): (128, 4, 3),
            (False, 1): (64, 4, 1),
            (True, 0): (128, 4, 3),
            (True, 1): (128, 4, 1),
            (True, 3): (64, 4, 3),
        },
        "slice_bwd_g": {
            (False, 0): (128, 4, 3),
            (False, 1): (128, 4, 1),
            (True, 0): (128, 4, 1),
            (True, 1): (256, 4, 1),
            (True, 3): (256, 4, 3),
        },
        "deslice_bwd_n": {
            (False, 0): (128, 4, 2),
            (False, 1): (64, 4, 1),
            (True, 0): (128, 4, 2),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 3),
        },
        "deslice_bwd_g": {
            (False, 0): (128, 4, 1),
            (False, 1): (128, 4, 1),
            (True, 0): (256, 8, 2),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 3),
        },
    },
    (64, 32): {
        "stats": {
            (False, 0): (128, 4, 1),
            (False, 1): (128, 4, 1),
            (True, 0): (128, 4, 1),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 3),
        },
        "slice_fwd_g": {
            (False, 0): (128, 8, 3),
            (False, 1): (128, 4, 1),
            (True, 0): (128, 4, 3),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 1),
        },
        "deslice_fwd_n": {
            (False, 0): (128, 4, 3),
            (False, 1): (64, 4, 1),
            (True, 0): (128, 4, 3),
            (True, 1): (64, 4, 1),
            (True, 3): (128, 4, 2),
        },
        "slice_bwd_n": {
            (False, 0): (64, 4, 2),
            (False, 1): (64, 4, 1),
            (True, 0): (64, 4, 2),
            (True, 1): (64, 4, 1),
            (True, 3): (64, 4, 3),
        },
        "slice_bwd_g": {
            (False, 0): (128, 4, 1),
            (False, 1): (128, 4, 1),
            (True, 0): (64, 4, 2),
            (True, 1): (128, 4, 1),
            (True, 3): (128, 4, 3),
        },
        "deslice_bwd_n": {
            (False, 0): (64, 4, 3),
            (False, 1): (64, 4, 1),
            (True, 0): (128, 8, 3),
            (True, 1): (64, 4, 1),
            (True, 3): (64, 4, 2),
        },
        "deslice_bwd_g": {
            (False, 0): (64, 4, 1),
            (False, 1): (128, 4, 1),
            (True, 0): (64, 4, 2),
            (True, 1): (128, 4, 1),
            (True, 3): (64, 4, 3),
        },
    },
}


def _cfg_blk(kernel, is16, dot, gb, dt):
    entry = _CFG_BLK.get((gb, dt), {}).get(kernel, {})
    # exact dot level first; bf16v (2) has no sweep of its own and takes the
    # tf32 entry, whose value dots also run on tensor cores. Below level 3
    # the math is fp32 whatever the input dtype (loads convert), so 16-bit
    # inputs may take the fp32 entry of the same level. Level 3 (all dots
    # bf16) never borrows a lower level: MMA-layout tiles are their own
    # world, and its absence means the single-tile bf16 table applies.
    # tf32x3 (4) has tf32's register profile (MMA operands, fp32 tiles) and
    # takes its entry until it has a sweep of its own.
    keys = [(is16, dot)]
    if dot in (2, 4):
        keys.append((is16, 1))
    if dot != 3:
        keys.append((is16, 0))
        if is16:
            keys += [(False, dot), (False, 1 if dot in (2, 4) else 0), (False, 0)]
    for key in keys:
        if key in entry:
            return entry[key]
    return None


# Which single-tile tuned table each blocked kernel borrows when _CFG_BLK
# has no entry for its (G_block, D_tile): the one whose register profile
# is closest (accumulator shape and number of dots).
_FAMILY = {
    "stats": "deslice_fwd",
    "slice_fwd_g": "slice_fwd",
    "deslice_fwd_n": "deslice_fwd",
    "slice_bwd_n": "slice_bwd",
    "slice_bwd_g": "slice_bwd",
    "deslice_bwd_n": "deslice_bwd",
    "deslice_bwd_g": "deslice_bwd",
}


def _launch_cfg(kernel, t, dot, gb, dt):
    """(BLOCK_N, num_warps, num_stages) for a blocked kernel."""
    cfg = _cfg_blk(kernel, t.dtype != torch.float32, dot, gb, dt)
    bn, warps, stages = cfg or _cfg(_FAMILY[kernel], t, dot, gb, dt)
    stages = _stages(dot, stages)
    if dot and warps > 4 and kernel.endswith(("_bwd_n", "_bwd_g")):
        # Triton 3.0 aborts the process (assert, not exception) compiling a
        # backward kernel with tensor-core dots on 8 warps: "mma -> mma
        # layout conversion is only supported on Ampere" (job 3262885,
        # slice_bwd_n bn64 w8 s1 bf16 at G_block=64; job 3264786, the
        # single-tile slice_bwd at G=128 under tf32x3). Forward kernels on 8
        # warps compiled and ran in the same sweeps.
        warps = 4
    return bn, warps, stages


def _consts(D, G, dt, gb, bn, dot, warps, stages):
    """The constexpr and launch keyword arguments of every blocked kernel."""
    return dict(D=D, DT=dt, GB=gb, BN=bn, DOT=dot, PAD_G=(G % gb != 0),
                PAD_D=(dt != D), num_warps=warps, num_stages=stages)


def _bh_dims(x_mid, weight):
    B, N, H, D = x_mid.shape
    return B, N, H, D, weight.shape[0]


def compute_stats(x_mid, weight, bias, tau, dot=0):
    """(B, H, 2, N) fp32 softmax statistics of the slice logits per point:
    [..., 0, :] the row max, [..., 1, :] the sum of exponentials."""
    B, N, H, D, G = _bh_dims(x_mid, weight)
    dt, gb = tiles(D, G)
    bn, warps, stages = _launch_cfg("stats", x_mid, dot, gb, dt)
    stats = torch.empty(B, H, 2, N, device=x_mid.device, dtype=torch.float32)
    _stats_kernel[(triton.cdiv(N, bn), B * H)](
        x_mid, weight.contiguous(), bias.contiguous(), tau.contiguous(), stats,
        N, G, H, *_strides(x_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return stats


def _slice_blk_impl(x_mid, fx_mid, weight, bias, tau, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    stats = compute_stats(x_mid, weight, bias, tau, dot)
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    bn, warps, stages = _launch_cfg("slice_fwd_g", x_mid, dot, gb, dt)
    P = _n_programs(N, B * H * ngb, bn)
    part_z = torch.empty(B * H * P, G, D, device=x_mid.device,
                         dtype=torch.float32)
    part_s = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _slice_fwd_g_kernel[(ngb, P, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, part_z, part_s,
        N, G, P, H, *_strides(x_mid), *_strides(fx_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return (part_z.view(B, H, P, G, D).sum(2), part_s.view(B, H, P, G).sum(2),
            stats)


def _slice_blk_bwd_impl(x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds,
                        dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    stats = stats.contiguous()
    dz_num = dz_num.contiguous().float()
    ds = ds.contiguous().float()
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    dxm = torch.empty_like(x_mid)
    dfx = torch.empty_like(fx_mid)
    delta = torch.empty(B, H, N, device=x_mid.device, dtype=torch.float32)
    bn, warps, stages = _launch_cfg("slice_bwd_n", x_mid, dot, gb, dt)
    nprog = triton.cdiv(N, bn)
    pdt = torch.empty(B * H * nprog, device=x_mid.device, dtype=torch.float32)
    _slice_bwd_n_kernel[(nprog, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds, dxm, dfx, delta,
        pdt, N, G, H, *_strides(x_mid), *_strides(fx_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    bn, warps, stages = _launch_cfg("slice_bwd_g", x_mid, dot, gb, dt)
    P = _n_programs(N, B * H * ngb, bn)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _slice_bwd_g_kernel[(ngb, P, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, delta, dz_num, ds,
        pdw, pdb,
        N, G, P, H, *_strides(x_mid), *_strides(fx_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return (dxm, dfx, pdw.sum(0), pdb.sum(0),
            pdt.view(B, H, nprog).sum(dim=(0, 2)))


def _deslice_blk_impl(x_mid, weight, bias, tau, tokens, stats, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    tokens = tokens.contiguous()
    if stats is None:
        stats = compute_stats(x_mid, weight, bias, tau, dot)
    else:
        stats = stats.contiguous()
    dt, gb = tiles(D, G)
    out = torch.empty(B, N, H, D, device=x_mid.device, dtype=x_mid.dtype)
    bn, warps, stages = _launch_cfg("deslice_fwd_n", x_mid, dot, gb, dt)
    _deslice_fwd_n_kernel[(triton.cdiv(N, bn), B * H)](
        x_mid, weight, bias, tau, tokens, stats, out,
        N, G, H, *_strides(x_mid), *_strides(out),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return out, stats


def _deslice_blk_bwd_impl(x_mid, weight, bias, tau, tokens, stats, d_out, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    tokens = tokens.contiguous()
    d_out = d_out.contiguous()
    stats = stats.contiguous()
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    dxm = torch.empty_like(x_mid)
    delta = torch.empty(B, H, N, device=x_mid.device, dtype=torch.float32)
    bn, warps, stages = _launch_cfg("deslice_bwd_n", x_mid, dot, gb, dt)
    nprog = triton.cdiv(N, bn)
    pdt = torch.empty(B * H * nprog, device=x_mid.device, dtype=torch.float32)
    _deslice_bwd_n_kernel[(nprog, B * H)](
        x_mid, weight, bias, tau, tokens, stats, d_out, dxm, delta, pdt,
        N, G, H, *_strides(x_mid), *_strides(d_out),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    bn, warps, stages = _launch_cfg("deslice_bwd_g", x_mid, dot, gb, dt)
    P = _n_programs(N, B * H * ngb, bn)
    pdtok = torch.empty(B * H * P, G, D, device=x_mid.device,
                        dtype=torch.float32)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _deslice_bwd_g_kernel[(ngb, P, B * H)](
        x_mid, weight, bias, tau, tokens, stats, delta, d_out,
        pdtok, pdw, pdb,
        N, G, P, H, *_strides(x_mid), *_strides(d_out),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return (dxm, pdtok.view(B, H, P, G, D).sum(2).to(tokens.dtype),
            pdw.sum(0), pdb.sum(0), pdt.view(B, H, nprog).sum(dim=(0, 2)))


# --------------------------------------------------------------------------- #
# custom ops — same contract as the single-tile ops in slice_ops: opaque to
# torch.compile, autograd-registered, fake-registered.
# --------------------------------------------------------------------------- #

torch.library.define(
    "flashslice::slice_blk",
    "(Tensor x_mid, Tensor fx_mid, Tensor weight, Tensor bias, Tensor tau, "
    "int dot) -> (Tensor, Tensor, Tensor)")
torch.library.define(
    "flashslice::slice_blk_bwd",
    "(Tensor x_mid, Tensor fx_mid, Tensor weight, Tensor bias, Tensor tau, "
    "Tensor stats, Tensor dz_num, Tensor ds, int dot) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)")
torch.library.define(
    "flashslice::deslice_blk",
    "(Tensor x_mid, Tensor weight, Tensor bias, Tensor tau, Tensor tokens, "
    "Tensor? stats, int dot) -> (Tensor, Tensor)")
torch.library.define(
    "flashslice::deslice_blk_bwd",
    "(Tensor x_mid, Tensor weight, Tensor bias, Tensor tau, Tensor tokens, "
    "Tensor stats, Tensor d_out, int dot) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)")

torch.library.impl("flashslice::slice_blk", "CUDA", _slice_blk_impl)
torch.library.impl("flashslice::slice_blk_bwd", "CUDA", _slice_blk_bwd_impl)
torch.library.impl("flashslice::deslice_blk", "CUDA", _deslice_blk_impl)
torch.library.impl("flashslice::deslice_blk_bwd", "CUDA", _deslice_blk_bwd_impl)


@torch.library.register_fake("flashslice::slice_blk")
def _(x_mid, fx_mid, weight, bias, tau, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    return (x_mid.new_empty((B, H, G, D), dtype=torch.float32),
            x_mid.new_empty((B, H, G), dtype=torch.float32),
            x_mid.new_empty((B, H, 2, N), dtype=torch.float32))


@torch.library.register_fake("flashslice::slice_blk_bwd")
def _(x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds, dot):
    return (torch.empty_like(x_mid), torch.empty_like(fx_mid),
            torch.empty_like(weight), torch.empty_like(bias),
            torch.empty_like(tau))


@torch.library.register_fake("flashslice::deslice_blk")
def _(x_mid, weight, bias, tau, tokens, stats, dot):
    B, N, H, D = x_mid.shape
    return (x_mid.new_empty((B, N, H, D)),
            x_mid.new_empty((B, H, 2, N), dtype=torch.float32))


@torch.library.register_fake("flashslice::deslice_blk_bwd")
def _(x_mid, weight, bias, tau, tokens, stats, d_out, dot):
    return (torch.empty_like(x_mid), torch.empty_like(tokens),
            torch.empty_like(weight), torch.empty_like(bias),
            torch.empty_like(tau))


def _slice_setup(ctx, inputs, output):
    x_mid, fx_mid, weight, bias, tau, dot = inputs
    ctx.save_for_backward(x_mid, fx_mid, weight, bias, tau, output[2])
    ctx.dot = dot


def _slice_grad(ctx, dz_num, ds, dstats):
    # The statistics are saved intermediates, not a differentiable output:
    # the Jacobian applied in the backward already accounts for the
    # normalization, so an incoming gradient on them (only possible by
    # bypassing fused_slice, which detaches them) is ignored.
    x_mid, fx_mid, weight, bias, tau, stats = ctx.saved_tensors
    if ds is None:
        ds = torch.zeros(dz_num.shape[:-1], device=dz_num.device,
                         dtype=torch.float32)
    dxm, dfx, dw, db, dtau = torch.ops.flashslice.slice_blk_bwd(
        x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds, ctx.dot)
    return dxm, dfx, dw, db, dtau, None


def _deslice_setup(ctx, inputs, output):
    x_mid, weight, bias, tau, tokens, stats, dot = inputs
    ctx.save_for_backward(x_mid, weight, bias, tau, tokens, output[1])
    ctx.dot = dot


def _deslice_grad(ctx, d_out, dstats):
    x_mid, weight, bias, tau, tokens, stats = ctx.saved_tensors
    dxm, dtok, dw, db, dtau = torch.ops.flashslice.deslice_blk_bwd(
        x_mid, weight, bias, tau, tokens, stats, d_out, ctx.dot)
    return dxm, dw, db, dtau, dtok, None, None


torch.library.register_autograd("flashslice::slice_blk", _slice_grad,
                                setup_context=_slice_setup)
torch.library.register_autograd("flashslice::deslice_blk", _deslice_grad,
                                setup_context=_deslice_setup)

__all__ = ["tiles", "set_block_g", "compute_stats"]
