# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""G-blocked slice/deslice kernels: any slice count, head width up to 256.

The single-tile kernels in ``slice_ops`` hold the whole slice axis G in one
tile, so the softmax over G is register-local and needs no bookkeeping — and
that is also what caps them at G <= 128. This module lifts the cap the way
FlashAttention-2's *backward* does, not the way its forward does:

* A small pass computes the per-point softmax statistics of the slice
  logits — the row max ``m[n]`` and the sum of exponentials ``l[n]`` — in
  one online pass over G in blocks of ``GB``; a deslice with no statistics
  in hand forms its output in that same pass and hands the statistics on.
  That is two floats per point and head, 2/G of the tensor the eager path
  stores, and it is kept for the backward.
* Every other kernel recomputes ``w[n, g] = exp(logit[n, g] - m[n]) / l[n]``
  one G-block at a time. The kernels that own points form the row sum they
  divide by themselves, from the very exponentials they use (it costs one
  reduction per block); the kernels that own slots take the saved ``l``.
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
the dominant weights); the point-owning kernels normalizing with a row sum
they form themselves, so that a row of w sums to one to the precision of
one summation whatever rounding the saved l carries (an online l used
directly put 3x eager's error on this gradient at N=17, G=256); and delta
computed from those same exponentials block by block in the N-owned
kernels, not from the algebraically equal fx . dfx + w @ ds. The last was
decisive — 3x eager's error at N=17, G=256 before, parity after — and the
other two are cheap insurance of the same kind.

Cost against the single-tile path at a shape both can serve: one extra read
of x_mid for the statistics pass (none when a deslice runs first: its online
pass yields them), one more recompute of the logits in each backward (the
N-owned kernels make two passes over G), and the G-owned kernels re-read
their inputs once per G-block. The block index is the
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

import os

import torch
import triton
import triton.language as tl

from .slice_ops import (_cfg, _dot, _dot_w, _n_programs, _reduce_parts, _stages,
                        _strides, _wb_layout)

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


_STATS_MODES = ("online", "two-pass")
_STATS_MODE = os.environ.get("FLASHSLICE_STATS_MODE", "").lower() or "online"


def set_stats_mode(mode):
    """How the per-point softmax statistics (m, l) are formed when no
    statistics are handed in: ``"online"`` (default) in one pass over G,
    with the deslice computing its output in the same pass; ``"two-pass"``
    with the max first and the sum against it second, the original form.
    Overrides FLASHSLICE_STATS_MODE. Both are held to the parity gate; the
    switch exists to attribute a difference, not to trade precision."""
    global _STATS_MODE
    if mode not in _STATS_MODES:
        raise ValueError("stats mode must be one of %s, got %r"
                         % (_STATS_MODES, mode))
    _STATS_MODE = mode


def stats_mode():
    return _STATS_MODE


# Whether the backward kernels that own points form the row sum they
# normalize with themselves (True) or take the saved l (False). Both are held
# to the parity gate; the switch attributes a difference and costs one
# reduction per G-block in pass 1 when on.
_OWN_L = (os.environ.get("FLASHSLICE_OWN_L", "1").lower() not in ("0", "false", "off"))


def set_own_row_sum(flag):
    global _OWN_L
    _OWN_L = bool(flag)


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
def _logits(xm, w_mat, bias, inv_tau, DOT: tl.constexpr):
    """(BN, GB) slice logits: bias before the temperature, as eager. The
    division by tau is a multiply by its reciprocal, formed once per program:
    an fp32 division is a multi-instruction sequence per element, the
    reciprocal costs one rounding of the logit."""
    return (_dot_w(xm, tl.trans(w_mat), DOT) + bias[None, :]) * inv_tau


@triton.jit
def _load_stats(STATS, bh64, N, offs_n64):
    """Row max m and exponential sum l of the softmax, (BN,) each. Rows past
    N read row N - 1 (see _load_vec): finite values every consumer masks."""
    base = STATS + bh64 * 2 * N
    m = _load_vec(base, offs_n64, N)
    l = _load_vec(base + N, offs_n64, N)
    return m, l


@triton.jit
def _expw(lg, m):
    """Unnormalized softmax rows exp(lg - m)."""
    return tl.exp(lg - m[:, None])


@triton.jit
def _weights(lg, m, inv_l):
    """Softmax rows from logits and statistics: exp(lg - m) * (1 / l)."""
    return tl.exp(lg - m[:, None]) * inv_l[:, None]


@triton.jit
def _stats_kernel(
    XM, W, BS, TAU, STATS,
    N, G, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """m[n] = max_g logit[n, g], l[n] = sum_g exp(logit[n, g] - m[n]), in one
    pass over G in FlashAttention's online form: l is rescaled by
    exp(m_old - m_new) whenever the row max moves. Stored as STATS[b, h, 0, n]
    and STATS[b, h, 1, n].

    Each rescale rounds l once more than a sum against the final max would,
    and that is why the kernels that own points (deslice_fwd_n, *_bwd_n)
    read only m from here and form the row sum they normalize with
    themselves, from the very exponentials they use: the temperature
    gradient is sum dl * logit with sum_g dl = -delta * (row sum of w - 1),
    so a row of w that does not sum to one to the precision of one
    summation becomes a bias scaled by the logits (3x eager's error at N=17,
    G=256 when an online l was used there). The kernels that own slots
    (slice_fwd_g, *_bwd_g) normalize with this l: for them a row factor of
    1 + O(ulp) is a relative perturbation of that row's contribution, not a
    cancellation. ``_stats_twopass_kernel`` is the max-then-sum form this
    replaces, kept behind ``set_stats_mode("two-pass")``."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    m = tl.full((BN,), float("-inf"), tl.float32)
    l = tl.zeros((BN,), tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, inv_tau, DOT)
        if PAD_G:
            lg = tl.where(gmask[None, :], lg, float("-inf"))
        m_new = tl.maximum(m, tl.max(lg, axis=1))
        e = _expw(lg, m_new)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        l = l * tl.exp(m - m_new) + tl.sum(e, axis=1)
        m = m_new
    base = STATS + bh.to(tl.int64) * 2 * N
    tl.store(base + offs_n64, m, mask=nmask)
    tl.store(base + N + offs_n64, l, mask=nmask)


@triton.jit
def _stats_twopass_kernel(
    XM, W, BS, TAU, STATS,
    N, G, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """The statistics in two passes over G: the max first, then the sum of
    exponentials against the final max, so that l is the same sum every
    consumer implicitly forms, to the precision of one summation. This was
    the only form until the point-owning kernels started forming their own
    row sums (see _stats_kernel); it costs one more logits pass and stays
    available through ``set_stats_mode("two-pass")``."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    m = tl.full((BN,), float("-inf"), tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, inv_tau, DOT)
        if PAD_G:
            lg = tl.where(gmask[None, :], lg, float("-inf"))
        m = tl.maximum(m, tl.max(lg, axis=1))
    l = tl.zeros((BN,), tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        e = _expw(_logits(xm, w_mat, bias, inv_tau, DOT), m)
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
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """Owns one G-block, streams over N: z_num[g] = sum_n w[n,g] fx[n],
    s[g] = sum_n w[n,g]. Per-program partials, reduced on the host."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    acc_z = tl.zeros((GB, DVT), dtype=tl.float32)
    acc_s = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        w = _weights(_logits(xm, w_mat, bias, inv_tau, DOT), m, 1.0 / l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        fx = _load_rows(FX, b, h, offs_n64, offs_v, nmask, vmask,
                        sfb, sfn, sfh, sfd, PAD_V)
        acc_z += _dot(tl.trans(w), fx, DOT)
        acc_s += tl.sum(w, axis=0)

    idx = (bh * P + pid).to(tl.int64)
    _store_part(PART_Z, acc_z, idx, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
    if PAD_G:
        tl.store(PART_S + idx * G + offs_g, acc_s, mask=gmask)
    else:
        tl.store(PART_S + idx * G + offs_g, acc_s)


@triton.jit
def _deslice_fwd_n_kernel(
    XM, W, BS, TAU, TOK, STATS, OUT,
    N, G, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """Owns one N-block, streams over G with the statistics in hand:
    out[n] = sum_g w[n,g] z'[g], accumulated unnormalized and scaled once at
    the end by 1/l. An output is a plain weighted sum, so the saved l serves
    it (a row factor of 1 + O(ulp) is a relative perturbation, not a
    cancellation); only the backward's Jacobian terms need the row sum formed
    in place (see _slice_bwd_n_kernel)."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    m, l = _load_stats(STATS, bh64, N, offs_n64)
    inv_l = 1.0 / l
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    acc = tl.zeros((BN, DVT), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        e = _expw(_logits(xm, w_mat, bias, inv_tau, DOT), m)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        acc += _dot(e, tok, DOT)
    _store_rows(OUT, acc * inv_l[:, None], b, h, offs_n64, offs_v, nmask, vmask,
                sob, son, soh, sod, PAD_V)


@triton.jit
def _deslice_fwd_online_kernel(
    XM, W, BS, TAU, TOK, STATS, OUT,
    N, G, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """Owns one N-block, streams over G with no statistics in hand:
    out[n] = sum_g w[n,g] z'[g] by online softmax (FlashAttention's forward:
    the accumulator and the row sum are rescaled when the row max moves),
    and the (m, l) it ends with are stored for the backward and for a tied
    slice. A deslice-first tied coupling runs on this kernel and skips the
    statistics pass; a slice-first one has its statistics from the slice
    and takes _deslice_fwd_n_kernel."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    m = tl.full((BN,), float("-inf"), tl.float32)
    l = tl.zeros((BN,), dtype=tl.float32)
    acc = tl.zeros((BN, DVT), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, inv_tau, DOT)
        if PAD_G:
            lg = tl.where(gmask[None, :], lg, float("-inf"))
        m_new = tl.maximum(m, tl.max(lg, axis=1))
        alpha = tl.exp(m - m_new)
        e = _expw(lg, m_new)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        l = l * alpha + tl.sum(e, axis=1)
        tok = _load_tok(TOK, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        acc = acc * alpha[:, None] + _dot(e, tok, DOT)
        m = m_new
    inv_l = 1.0 / l
    _store_rows(OUT, acc * inv_l[:, None], b, h, offs_n64, offs_v, nmask, vmask,
                sob, son, soh, sod, PAD_V)
    base = STATS + bh64 * 2 * N
    tl.store(base + offs_n64, m, mask=nmask)
    tl.store(base + N + offs_n64, l, mask=nmask)


@triton.jit
def _slice_bwd_n_kernel(
    XM, FX, W, BS, TAU, STATS, DZN, DS,
    DXM, DFX, DELTA, PDT,
    N, G, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
    OWN_L: tl.constexpr,
):
    """Owns one N-block. Pass 1 over G, unnormalized: dfx = e @ dz_num and
    the Jacobian numerator sum_g (e dw), with the row sum l = sum_g e formed
    alongside when OWN_L (else the saved l is taken); everything is scaled by
    1/l once. Pass 2: dxm, and the temperature gradient.

    delta = (sum_g e_g dw_g) / l and dl_g = (e_g dw_g - e_g delta) / l are
    built from the same rounded products e_g dw_g, so that sum_g dl vanishes
    to the precision of one summation, as it does in the single-tile
    kernels. Normalizing w_g = e_g / l first and forming delta from w_g dw_g
    while dl uses w_g (dw_g - delta) puts a second, independently rounded
    product into the identity, and the temperature gradient -- sum dl *
    logit, a cancellation -- showed it as 1.5-1.9x eager's error (N=17 and
    N=4097, G=256 and 2048); the algebraically equal delta = fx . dfx +
    w @ ds (one dot cheaper) had shown 3x. Every other gradient is
    indifferent to either.

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
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    fx = _load_rows(FX, b, h, offs_n64, offs_v, nmask, vmask, sfb, sfn, sfh, sfd,
                    PAD_V)
    m, l_saved = _load_stats(STATS, bh64, N, offs_n64)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    acc_dfx = tl.zeros((BN, DVT), dtype=tl.float32)
    sdw = tl.zeros((BN,), dtype=tl.float32)
    l = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        e = _expw(_logits(xm, w_mat, bias, inv_tau, DOT), m)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        dzn = _load_tok(DZN, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        ds = _load_vec(DS + bh64 * G, offs_g, G)
        acc_dfx += _dot(e, dzn, DOT)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        sdw += tl.sum(e * dw, axis=1)
        if OWN_L:
            l += tl.sum(e, axis=1)
    if OWN_L:
        inv_l = 1.0 / l
    else:
        inv_l = 1.0 / l_saved
    delta = sdw * inv_l
    _store_rows(DFX, acc_dfx * inv_l[:, None], b, h, offs_n64, offs_v, nmask, vmask,
                sfb, sfn, sfh, sfd, PAD_V)
    tl.store(DELTA + bh64 * N + offs_n64, delta, mask=nmask)

    acc_dxm = tl.zeros((BN, DT), dtype=tl.float32)
    row_dt = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, inv_tau, DOT)
        e = _expw(lg, m)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        dzn = _load_tok(DZN, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        ds = _load_vec(DS + bh64 * G, offs_g, G)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        dl = (e * dw - e * delta[:, None]) * inv_l
        acc_dxm += _dot(dl * inv_tau, w_mat, DOT)
        row_dt += tl.sum(dl * lg, axis=1)
    _store_rows(DXM, acc_dxm, b, h, offs_n64, offs_d, nmask, dmask,
                sxb, sxn, sxh, sxd, PAD_D)
    tl.store(PDT + bh64 * tl.num_programs(0) + pid,
             -inv_tau * tl.sum(tl.where(nmask, row_dt, 0.0), axis=0))


@triton.jit
def _slice_bwd_g_kernel(
    XM, FX, W, BS, TAU, STATS, DELTA, DZN, DS,
    PDW, PDB,
    N, G, P, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """Owns one G-block, streams over N: partial dW and db."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)
    dzn = _load_tok(DZN, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
    ds = _load_vec(DS + bh64 * G, offs_g, G)

    acc_dw = tl.zeros((GB, DT), dtype=tl.float32)
    acc_db = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        fx = _load_rows(FX, b, h, offs_n64, offs_v, nmask, vmask,
                        sfb, sfn, sfh, sfd, PAD_V)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        delta = _load_vec(DELTA + bh64 * N, offs_n64, N)
        w = _weights(_logits(xm, w_mat, bias, inv_tau, DOT), m, 1.0 / l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        dlr = w * (dw - delta[:, None]) * inv_tau
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
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
    OWN_L: tl.constexpr,
):
    """Owns one N-block. Pass 1 over G, unnormalized: the Jacobian numerator
    sum_g e (dout . z'[g]) and, when OWN_L, the row sum l = sum_g e (delta is
    recomputed rather than read back from a rounded `out`). Pass 2: dxm and
    the temperature gradient from the same products (row by row over all of
    G, for the reasons given on _slice_bwd_n_kernel)."""
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_n = pid * BN + tl.arange(0, BN)
    nmask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)
    xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask, sxb, sxn, sxh, sxd,
                    PAD_D)
    dout = _load_rows(DOUT, b, h, offs_n64, offs_v, nmask, vmask, sob, son, soh,
                      sod, PAD_V)
    m, l_saved = _load_stats(STATS, bh64, N, offs_n64)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)

    sdw = tl.zeros((BN,), dtype=tl.float32)
    l = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        e = _expw(_logits(xm, w_mat, bias, inv_tau, DOT), m)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        sdw += tl.sum(e * _dot(dout, tl.trans(tok), DOT), axis=1)
        if OWN_L:
            l += tl.sum(e, axis=1)
    if OWN_L:
        inv_l = 1.0 / l
    else:
        inv_l = 1.0 / l_saved
    delta = sdw * inv_l
    tl.store(DELTA + bh64 * N + offs_n64, delta, mask=nmask)

    acc_dxm = tl.zeros((BN, DT), dtype=tl.float32)
    row_dt = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GB):
        offs_g = g0 + tl.arange(0, GB)
        gmask = offs_g < G
        w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
        lg = _logits(xm, w_mat, bias, inv_tau, DOT)
        e = _expw(lg, m)
        if PAD_G:
            e = tl.where(gmask[None, :], e, 0.0)
        tok = _load_tok(TOK, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)
        dw = _dot(dout, tl.trans(tok), DOT)
        dl = (e * dw - e * delta[:, None]) * inv_l
        acc_dxm += _dot(dl * inv_tau, w_mat, DOT)
        row_dt += tl.sum(dl * lg, axis=1)
    _store_rows(DXM, acc_dxm, b, h, offs_n64, offs_d, nmask, dmask,
                sxb, sxn, sxh, sxd, PAD_D)
    tl.store(PDT + bh64 * tl.num_programs(0) + pid,
             -inv_tau * tl.sum(tl.where(nmask, row_dt, 0.0), axis=0))


@triton.jit
def _deslice_bwd_g_kernel(
    XM, W, BS, TAU, TOK, STATS, DELTA, DOUT,
    PDTOK, PDW, PDB,
    N, G, P, H,
    swb, swh, sbb, sbh,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, DT: tl.constexpr, GB: tl.constexpr, BN: tl.constexpr,
    DOT: tl.constexpr, PAD_G: tl.constexpr, PAD_D: tl.constexpr,
    DV: tl.constexpr, DVT: tl.constexpr, PAD_V: tl.constexpr,
):
    """Owns one G-block, streams over N: partial dz', dW and db."""
    gblk = tl.program_id(0)
    pid = tl.program_id(1)
    bh = tl.program_id(2)
    b = bh // H
    h = bh % H
    W = W + b.to(tl.int64) * swb + h * swh
    BS = BS + b.to(tl.int64) * sbb + h * sbh
    bh64 = bh.to(tl.int64)
    offs_d = tl.arange(0, DT)
    dmask = offs_d < D
    offs_v = tl.arange(0, DVT)
    vmask = offs_v < DV
    offs_g = gblk * GB + tl.arange(0, GB)
    gmask = offs_g < G
    w_mat, bias = _load_wb(W, BS, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    inv_tau = 1.0 / tl.load(TAU + h).to(tl.float32)
    tok = _load_tok(TOK, bh64, offs_g, offs_v, gmask, vmask, G, DV, PAD_G, PAD_V)

    acc_dtok = tl.zeros((GB, DVT), dtype=tl.float32)
    acc_dw = tl.zeros((GB, DT), dtype=tl.float32)
    acc_db = tl.zeros((GB,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        xm = _load_rows(XM, b, h, offs_n64, offs_d, nmask, dmask,
                        sxb, sxn, sxh, sxd, PAD_D)
        dout = _load_rows(DOUT, b, h, offs_n64, offs_v, nmask, vmask,
                          sob, son, soh, sod, PAD_V)
        m, l = _load_stats(STATS, bh64, N, offs_n64)
        delta = _load_vec(DELTA + bh64 * N, offs_n64, N)
        w = _weights(_logits(xm, w_mat, bias, inv_tau, DOT), m, 1.0 / l)
        if PAD_G:
            w = tl.where(nmask[:, None] & gmask[None, :], w, 0.0)
        else:
            w = tl.where(nmask[:, None], w, 0.0)
        acc_dtok += _dot(tl.trans(w), dout, DOT)
        dw = _dot(dout, tl.trans(tok), DOT)
        dlr = w * (dw - delta[:, None]) * inv_tau
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)

    idx = (bh * P + pid).to(tl.int64)
    _store_part(PDTOK, acc_dtok, idx, offs_g, offs_v, gmask, vmask, G, DV,
                PAD_G, PAD_V)
    _store_part(PDW, acc_dw, idx, offs_g, offs_d, gmask, dmask, G, D, PAD_G, PAD_D)
    if PAD_G:
        tl.store(PDB + idx * G + offs_g, acc_db, mask=gmask)
    else:
        tl.store(PDB + idx * G + offs_g, acc_db)


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #

# (BLOCK_N, num_warps, num_stages) per (G_block, D_tile) or
# (G_block, D_tile, DV_tile) -> kernel -> (input-is-16bit, dot level), from
# bench_kernels.py --family blocked sweeps on one GH200 at N=262k (see the
# job numbers next to each table). A key with the value width is looked up
# first, then the two-element key for any value width. The
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


def _cfg_blk(kernel, is16, dot, gb, dt, dvt=None):
    # (G_block, D_tile, DV_tile) first -- the value width sets the shape of
    # the value accumulators -- then (G_block, D_tile) for any value width.
    entry = {}
    for key in ((gb, dt, dvt), (gb, dt)):
        entry = _CFG_BLK.get(key, {}).get(kernel)
        if entry:
            break
    entry = entry or {}
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
    "deslice_fwd_online": "deslice_fwd",
    "slice_bwd_n": "slice_bwd",
    "slice_bwd_g": "slice_bwd",
    "deslice_bwd_n": "deslice_bwd",
    "deslice_bwd_g": "deslice_bwd",
}


def _launch_cfg(kernel, t, dot, gb, dt, dvt=None):
    """(BLOCK_N, num_warps, num_stages) for a blocked kernel."""
    cfg = _cfg_blk(kernel, t.dtype != torch.float32, dot, gb, dt, dvt)
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


def _consts(D, G, dt, gb, bn, dot, warps, stages, dv=None):
    """The constexpr and launch keyword arguments of every blocked kernel.
    ``dv`` is the value width (fx_mid, tokens, out); None means it equals the
    logits width D of x_mid and the weight."""
    dv = D if dv is None else dv
    dvt = _pow2_at_least_16(dv)
    return dict(D=D, DT=dt, GB=gb, BN=bn, DOT=dot, PAD_G=(G % gb != 0),
                PAD_D=(dt != D), DV=dv, DVT=dvt, PAD_V=(dvt != dv),
                num_warps=warps, num_stages=stages)


def _bh_dims(x_mid, weight):
    B, N, H, D = x_mid.shape
    return B, N, H, D, weight.shape[-2]


def compute_stats(x_mid, weight, bias, tau, dot=0):
    """(B, H, 2, N) fp32 softmax statistics of the slice logits per point:
    [..., 0, :] the row max, [..., 1, :] the sum of exponentials."""
    B, N, H, D, G = _bh_dims(x_mid, weight)
    dt, gb = tiles(D, G)
    bn, warps, stages = _launch_cfg("stats", x_mid, dot, gb, dt)
    stats = torch.empty(B, H, 2, N, device=x_mid.device, dtype=torch.float32)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    wb = _wb_layout(weight, bias, B, H)
    kern = _stats_kernel if _STATS_MODE == "online" else _stats_twopass_kernel
    kern[(triton.cdiv(N, bn), B * H)](
        x_mid, weight, bias, tau, stats,
        N, G, H, *wb, *_strides(x_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages))
    return stats


def _slice_blk_impl(x_mid, fx_mid, weight, bias, tau, stats, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    wb = _wb_layout(weight, bias, B, H)
    if stats is None:
        stats = compute_stats(x_mid, weight, bias, tau, dot)
    else:
        stats = stats.contiguous()
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    DV = fx_mid.shape[3]
    bn, warps, stages = _launch_cfg("slice_fwd_g", x_mid, dot, gb, dt,
                                    _pow2_at_least_16(DV))
    P = _n_programs(N, B * H * ngb, bn)
    part_z = torch.empty(B * H * P, G, DV, device=x_mid.device,
                         dtype=torch.float32)
    part_s = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _slice_fwd_g_kernel[(ngb, P, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, part_z, part_s,
        N, G, P, H, *wb, *_strides(x_mid), *_strides(fx_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    return (part_z.view(B, H, P, G, DV).sum(2), part_s.view(B, H, P, G).sum(2),
            stats)


def _slice_blk_bwd_impl(x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds,
                        dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    wb = _wb_layout(weight, bias, B, H)
    stats = stats.contiguous()
    dz_num = dz_num.contiguous().float()
    ds = ds.contiguous().float()
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    dxm = torch.empty_like(x_mid)
    dfx = torch.empty_like(fx_mid)
    delta = torch.empty(B, H, N, device=x_mid.device, dtype=torch.float32)
    DV = fx_mid.shape[3]
    dvt = _pow2_at_least_16(DV)
    bn, warps, stages = _launch_cfg("slice_bwd_n", x_mid, dot, gb, dt, dvt)
    nprog = triton.cdiv(N, bn)
    pdt = torch.empty(B * H * nprog, device=x_mid.device, dtype=torch.float32)
    _slice_bwd_n_kernel[(nprog, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, dz_num, ds, dxm, dfx, delta,
        pdt, N, G, H, *wb, *_strides(x_mid), *_strides(fx_mid),
        OWN_L=_OWN_L, **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    bn, warps, stages = _launch_cfg("slice_bwd_g", x_mid, dot, gb, dt, dvt)
    P = _n_programs(N, B * H * ngb, bn)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _slice_bwd_g_kernel[(ngb, P, B * H)](
        x_mid, fx_mid, weight, bias, tau, stats, delta, dz_num, ds,
        pdw, pdb,
        N, G, P, H, *wb, *_strides(x_mid), *_strides(fx_mid),
        **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    dw, db = _reduce_parts(pdw, pdb, B, H, P, weight, bias)
    return dxm, dfx, dw, db, pdt.view(B, H, nprog).sum(dim=(0, 2))


def _deslice_blk_impl(x_mid, weight, bias, tau, tokens, stats, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    wb = _wb_layout(weight, bias, B, H)
    tokens = tokens.contiguous()
    dt, gb = tiles(D, G)
    DV = tokens.shape[3]
    dvt = _pow2_at_least_16(DV)
    out = torch.empty(B, N, H, DV, device=x_mid.device, dtype=x_mid.dtype)
    if stats is None and _STATS_MODE == "online":
        # No statistics in hand: one online pass forms out and (m, l) both.
        stats = torch.empty(B, H, 2, N, device=x_mid.device, dtype=torch.float32)
        bn, warps, stages = _launch_cfg("deslice_fwd_online", x_mid, dot, gb, dt, dvt)
        _deslice_fwd_online_kernel[(triton.cdiv(N, bn), B * H)](
            x_mid, weight, bias, tau, tokens, stats, out,
            N, G, H, *wb, *_strides(x_mid), *_strides(out),
            **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
        return out, stats
    if stats is None:
        stats = compute_stats(x_mid, weight, bias, tau, dot)
    else:
        stats = stats.contiguous()
    bn, warps, stages = _launch_cfg("deslice_fwd_n", x_mid, dot, gb, dt, dvt)
    _deslice_fwd_n_kernel[(triton.cdiv(N, bn), B * H)](
        x_mid, weight, bias, tau, tokens, stats, out,
        N, G, H, *wb, *_strides(x_mid), *_strides(out),
        **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    return out, stats


def _deslice_blk_bwd_impl(x_mid, weight, bias, tau, tokens, stats, d_out, dot):
    B, N, H, D, G = _bh_dims(x_mid, weight)
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    wb = _wb_layout(weight, bias, B, H)
    tokens = tokens.contiguous()
    d_out = d_out.contiguous()
    stats = stats.contiguous()
    dt, gb = tiles(D, G)
    ngb = triton.cdiv(G, gb)
    dxm = torch.empty_like(x_mid)
    delta = torch.empty(B, H, N, device=x_mid.device, dtype=torch.float32)
    DV = tokens.shape[3]
    dvt = _pow2_at_least_16(DV)
    bn, warps, stages = _launch_cfg("deslice_bwd_n", x_mid, dot, gb, dt, dvt)
    nprog = triton.cdiv(N, bn)
    pdt = torch.empty(B * H * nprog, device=x_mid.device, dtype=torch.float32)
    _deslice_bwd_n_kernel[(nprog, B * H)](
        x_mid, weight, bias, tau, tokens, stats, d_out, dxm, delta, pdt,
        N, G, H, *wb, *_strides(x_mid), *_strides(d_out),
        OWN_L=_OWN_L, **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    bn, warps, stages = _launch_cfg("deslice_bwd_g", x_mid, dot, gb, dt, dvt)
    P = _n_programs(N, B * H * ngb, bn)
    pdtok = torch.empty(B * H * P, G, DV, device=x_mid.device,
                        dtype=torch.float32)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _deslice_bwd_g_kernel[(ngb, P, B * H)](
        x_mid, weight, bias, tau, tokens, stats, delta, d_out,
        pdtok, pdw, pdb,
        N, G, P, H, *wb, *_strides(x_mid), *_strides(d_out),
        **_consts(D, G, dt, gb, bn, dot, warps, stages, DV))
    dw, db = _reduce_parts(pdw, pdb, B, H, P, weight, bias)
    return (dxm, pdtok.view(B, H, P, G, DV).sum(2).to(tokens.dtype),
            dw, db, pdt.view(B, H, nprog).sum(dim=(0, 2)))


# --------------------------------------------------------------------------- #
# custom ops — same contract as the single-tile ops in slice_ops: opaque to
# torch.compile, autograd-registered, fake-registered.
# --------------------------------------------------------------------------- #

torch.library.define(
    "flashslice::slice_blk",
    "(Tensor x_mid, Tensor fx_mid, Tensor weight, Tensor bias, Tensor tau, "
    "Tensor? stats, int dot) -> (Tensor, Tensor, Tensor)")
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
def _(x_mid, fx_mid, weight, bias, tau, stats, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[-2]
    return (x_mid.new_empty((B, H, G, fx_mid.shape[3]), dtype=torch.float32),
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
    return (x_mid.new_empty((B, N, H, tokens.shape[3])),
            x_mid.new_empty((B, H, 2, N), dtype=torch.float32))


@torch.library.register_fake("flashslice::deslice_blk_bwd")
def _(x_mid, weight, bias, tau, tokens, stats, d_out, dot):
    return (torch.empty_like(x_mid), torch.empty_like(tokens),
            torch.empty_like(weight), torch.empty_like(bias),
            torch.empty_like(tau))


def _slice_setup(ctx, inputs, output):
    x_mid, fx_mid, weight, bias, tau, stats, dot = inputs
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
    return dxm, dfx, dw, db, dtau, None, None


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

__all__ = ["tiles", "set_block_g", "compute_stats", "set_stats_mode", "stats_mode",
           "set_own_row_sum"]
