"""Fused Triton kernels for Transolver's slice/deslice pipeline ("FlashSlice").

Replaces the bandwidth-bound middle of ``Physics_Attention_Irregular_Mesh``
with two streaming passes over the N points, FlashAttention-style:

  slice:   w = softmax_G((x_mid @ W.T + b) / tau)   per point, recomputed on
           the fly; tokens z_num[g,:] = sum_n w[n,g] fx_mid[n,:],
           s[g] = sum_n w[n,g].  w (B,H,N,G) is never written to memory.
  deslice: out[n,:] = sum_g w[n,g] z'[g,:], w recomputed again.

Backward recomputes w a third/fourth time instead of loading it (the
FlashAttention recompute trade), applies the softmax Jacobian per point, and
accumulates the tiny parameter gradients (dW, db, dtau) and token gradients
in per-program fp32 partial buffers reduced with a deterministic ``.sum()``
— no atomics anywhere.

Exactness: identical math to the eager module (bias before the temperature
division, softmax over G, no temperature clamp) with fp32 accumulation.
Dot precision defaults to ieee fp32, so differences vs eager are fp
summation-order only (cuBLAS also serves the skinny slice einsums with fp32
kernels even under ``allow_tf32``). ``CAMELEON_FUSED_SLICE_DOT`` (or
``set_dot_mode``) opts into faster tensor-core dots: ``tf32`` (value dots
tf32, ~5e-4 output noise — between exact fp32 and the bf16 class), ``bf16v``
(value dots bf16, 16-bit inputs only) or ``bf16`` (all dots bf16 — the same
precision class as the eager bf16-autocast einsums). Accumulation stays fp32
in every mode; below the full-bf16 level the slice-weight logits dot stays
ieee because the softmax Jacobian amplifies any noise in w.

Layout: kernels read x_mid / fx_mid in the natural GEMM output layout
(B, N, H, D) via strides — the eager permute+contiguous copies disappear.
Constraints: D and G must be powers of two in [16, 128] (paper: D=32, G=32),
because each axis is held in a single tile. Outside that range
``Physics_Attention_Irregular_Mesh`` warns and runs the eager path instead
(``unsupported_dims``); inside it, tiles are tuned per G (``_CFG``).
"""

import os

import torch
import triton
import triton.language as tl

_MAX_PROGRAMS = 512  # target total persistent programs across the (B*H) grid axis

# (BLOCK_N, num_warps, num_stages) per G -> (kernel, input-is-16bit), from the
# GH200 sweeps of bench_kernels.py. G is a tiling axis, not just a shape: the
# whole slice axis lives in one tile, so the tile budget is BLOCK_N x G and the
# winner moves with G. Winners also differ by dtype: e.g. (256,4,2) is best for
# bf16 slice_fwd at G=32 but pathological for fp32 (job 3084301's ieee
# regression), whose winner is stages=1.
# Using the G=32 tiles at another G is not a small loss: the BLOCK_N x G
# accumulators spill to local memory, costing a median 2.4x at G=64 and 5.3x at
# G=128 (worst single kernel 40x / 80x), and leaving ~9% on the table at G=16.
#   G=32:  job 3084253 (+ 3088100 for the bf16-dot tier) — frozen; every number
#          reported in the study was measured with these tiles, and a fresh
#          sweep (3095916) reproduces 10 of its 12 entries exactly, the other
#          two within 2.3%.
#   G=16/64/128: jobs 3095933 / 3095917 / 3095918, one sweep each, winner =
#          lowest mean slowdown vs the per-N best over N = 262k and 1M.
_CFG = {
    16: {
        ("slice_fwd", False): (256, 4, 2),
        ("slice_fwd", True): (128, 4, 2),
        ("deslice_fwd", False): (256, 4, 1),
        ("deslice_fwd", True): (256, 4, 1),
        ("slice_bwd", False): (256, 4, 2),
        ("slice_bwd", True): (256, 4, 1),
        ("deslice_bwd", False): (128, 4, 1),
        ("deslice_bwd", True): (64, 4, 2),
    },
    32: {
        ("slice_fwd", False): (256, 4, 1),
        ("slice_fwd", True): (256, 4, 2),
        ("deslice_fwd", False): (128, 4, 1),
        ("deslice_fwd", True): (128, 4, 1),
        ("slice_bwd", False): (128, 4, 1),
        ("slice_bwd", True): (128, 4, 1),
        ("deslice_bwd", False): (128, 4, 2),
        ("deslice_bwd", True): (64, 4, 2),
    },
    64: {
        ("slice_fwd", False): (128, 4, 1),
        ("slice_fwd", True): (128, 4, 2),
        ("deslice_fwd", False): (128, 8, 1),
        ("deslice_fwd", True): (128, 4, 3),
        ("slice_bwd", False): (64, 4, 1),
        ("slice_bwd", True): (64, 4, 1),
        ("deslice_bwd", False): (64, 4, 2),
        ("deslice_bwd", True): (64, 4, 1),
    },
    128: {
        ("slice_fwd", False): (64, 4, 3),
        ("slice_fwd", True): (64, 4, 2),
        ("deslice_fwd", False): (64, 4, 1),
        ("deslice_fwd", True): (64, 4, 2),
        ("slice_bwd", False): (64, 8, 2),
        ("slice_bwd", True): (64, 8, 1),
        ("deslice_bwd", False): (64, 8, 2),
        ("deslice_bwd", True): (64, 8, 1),
    },
}


# bf16-dot (level 3) winners differ again — faster dots shift the optimum
# and stages>1 is safe there (same jobs as above).
_CFG_BF16 = {
    16: {
        "slice_fwd": (128, 4, 3),
        "deslice_fwd": (256, 4, 3),
        "slice_bwd": (64, 4, 3),
        "deslice_bwd": (128, 4, 2),
    },
    32: {
        "slice_fwd": (256, 4, 3),
        "deslice_fwd": (64, 4, 1),
        "slice_bwd": (64, 4, 3),
        "deslice_bwd": (64, 4, 3),
    },
    64: {
        "slice_fwd": (128, 4, 3),
        "deslice_fwd": (128, 8, 2),
        "slice_bwd": (64, 4, 1),
        "deslice_bwd": (64, 4, 2),
    },
    128: {
        "slice_fwd": (64, 4, 2),
        "deslice_fwd": (64, 4, 3),
        "slice_bwd": (64, 4, 1),
        "deslice_bwd": (64, 4, 1),
    },
}


def _cfg(name, t, dot=0, g=32, d=32):
    table = _CFG_BF16 if dot == 3 else _CFG
    key = name if dot == 3 else (name, t.dtype != torch.float32)
    # G=32 tiles are the fallback for any (G, tier) the sweep does not cover
    bn, warps, stages = table.get(g, {}).get(key) or table[32][key]
    if d != 32:
        # Tiles were swept at the paper head width D=32; keep BLOCK_N*D roughly
        # constant so a wide head does not blow the register budget. Heuristic,
        # not measured (D=32 in every experiment of the study).
        bn = max(16, min(256, bn * 32 // d))
    if dot and g <= 16 and warps > 4:
        # Triton 3.0 aborts (assert, not exception) when it splits a
        # tensor-core dot of N=16 across 8 warps: per-warp MMA N=8.
        warps = 4
    return bn, warps, stages

_DOT_ENV = os.environ.get("CAMELEON_FUSED_SLICE_DOT", "").lower()

# 0=ieee fp32, 1=tf32 value dots, 2=bf16 value dots, 3=all dots bf16.
# Accumulation is fp32 in every mode. The logits dot (w computation) stays
# ieee below level 3 — the softmax Jacobian amplifies noise in w.
_DOT_LEVELS = {"": 0, "ieee": 0, "tf32": 1, "bf16v": 2, "bf16": 3}


def set_dot_mode(mode):
    """'ieee' (default), 'tf32', 'bf16v' or 'bf16' — overrides
    CAMELEON_FUSED_SLICE_DOT. Explicit opt-in only; not tied to torch's
    allow_tf32 (cuBLAS serves the skinny eager slice einsums with fp32
    kernels even when that flag is set)."""
    global _DOT_ENV
    _DOT_ENV = mode


def _dot_mode(t):
    lvl = _DOT_LEVELS[_DOT_ENV]
    if lvl >= 2 and t.dtype == torch.float32:
        return 1  # bf16 dots only for 16-bit inputs; fp32 falls back to tf32
    return lvl


def unsupported_dims(d, g):
    """Why the fused kernels cannot serve these dims, or None if they can.

    Both the head width D and the slice count G are held in a single tile
    (there is no blocking over either axis — that is what lets the softmax
    over G stay register-local and avoids FlashAttention's online-softmax
    bookkeeping), and `tl.dot` needs at least 16 per dimension. So each must
    be a power of two in [16, 128]. G=8 and G=256 — the ends of the
    slice-count sweep — fall outside and run on the eager path.
    """
    for v, name in ((d, "dim_head"), (g, "slice_num")):
        if v < 16 or v > 128 or (v & (v - 1)) != 0:
            return ("%s=%d is outside the fused kernels' supported range "
                    "(powers of two in [16, 128])" % (name, v))
    return None


def _check_dims(d, g):
    why = unsupported_dims(d, g)
    if why:
        raise ValueError("fused_slice: " + why)


def _n_programs(n, bh, block_n):
    return max(1, min(triton.cdiv(n, block_n), max(8, _MAX_PROGRAMS // max(bh, 1))))


def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2), t.stride(3)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #

@triton.jit
def _dot(a, b, DOT: tl.constexpr):
    """Value/gradient dot at the requested precision; fp32 accumulation."""
    if DOT >= 2:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="tf32" if DOT == 1 else "ieee")


@triton.jit
def _dot_w(a, b, DOT: tl.constexpr):
    """Logits dot for w: ieee unless full-bf16 mode (level 3)."""
    if DOT == 3:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a, b, input_precision="ieee")

@triton.jit
def _slice_fwd_kernel(
    XM, FX, W, BS, TAU, PART_Z, PART_S,
    N, P, H,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)

    acc_z = tl.zeros((G, D), dtype=tl.float32)
    acc_s = tl.zeros((G,), dtype=tl.float32)
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)
        fx = tl.load(FX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                     + offs_d[None, :] * sfd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        acc_z += _dot(tl.trans(w), fx, DOT)
        acc_s += tl.sum(w, axis=0)

    idx = bh * P + pid
    tl.store(PART_Z + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_z)
    tl.store(PART_S + idx * G + offs_g, acc_s)


@triton.jit
def _deslice_fwd_kernel(
    XM, W, BS, TAU, TOK, OUT,
    N, H,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    tok = tl.load(TOK + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)

    offs_n = pid * BN + tl.arange(0, BN)
    mask = offs_n < N
    offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
    xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 mask=mask[:, None], other=0.0).to(tl.float32)
    logits = _dot_w(xm, tl.trans(w_mat), DOT)
    logits = (logits + bias[None, :]) / tau
    m = tl.max(logits, axis=1)
    e = tl.exp(logits - m[:, None])
    w = e / tl.sum(e, axis=1)[:, None]
    out = _dot(w, tok, DOT)
    tl.store(OUT + b * sob + h * soh + offs_n64[:, None] * son
             + offs_d[None, :] * sod,
             out.to(OUT.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _slice_bwd_kernel(
    XM, FX, W, BS, TAU, DZN, DS,
    DXM, DFX, PDW, PDB, PDT,
    N, P, H,
    sxb, sxn, sxh, sxd,
    sfb, sfn, sfh, sfd,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    dzn = tl.load(DZN + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)
    ds = tl.load(DS + bh * G + offs_g).to(tl.float32)

    acc_dw = tl.zeros((G, D), dtype=tl.float32)
    acc_db = tl.zeros((G,), dtype=tl.float32)
    acc_dt = 0.0
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)

        fx = tl.load(FX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                     + offs_d[None, :] * sfd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        # d fx_mid = w @ dz_num
        dfx = _dot(w, dzn, DOT)
        tl.store(DFX + b * sfb + h * sfh + offs_n64[:, None] * sfn
                 + offs_d[None, :] * sfd,
                 dfx.to(DFX.dtype.element_ty), mask=mask[:, None])
        # d w, softmax Jacobian, d logits (pre-division)
        dw = _dot(fx, tl.trans(dzn), DOT) + ds[None, :]
        gsum = tl.sum(dw * w, axis=1)
        dl = w * (dw - gsum[:, None])
        dlr = dl / tau
        dxm = _dot(dlr, w_mat, DOT)
        tl.store(DXM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 dxm.to(DXM.dtype.element_ty), mask=mask[:, None])
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)
        acc_dt += tl.sum(tl.sum(dl * (-logits / tau), axis=1), axis=0)

    idx = bh * P + pid
    tl.store(PDW + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dw)
    tl.store(PDB + idx * G + offs_g, acc_db)
    tl.store(PDT + idx, acc_dt)


@triton.jit
def _deslice_bwd_kernel(
    XM, W, BS, TAU, TOK, DOUT,
    DXM, PDTOK, PDW, PDB, PDT,
    N, P, H,
    sxb, sxn, sxh, sxd,
    sob, son, soh, sod,
    D: tl.constexpr, G: tl.constexpr, BN: tl.constexpr, DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, G)

    w_mat = tl.load(W + offs_g[:, None] * D + offs_d[None, :]).to(tl.float32)
    bias = tl.load(BS + offs_g).to(tl.float32)
    tau = tl.load(TAU + h).to(tl.float32)
    tok = tl.load(TOK + bh * G * D + offs_g[:, None] * D
                  + offs_d[None, :]).to(tl.float32)

    acc_dtok = tl.zeros((G, D), dtype=tl.float32)
    acc_dw = tl.zeros((G, D), dtype=tl.float32)
    acc_db = tl.zeros((G,), dtype=tl.float32)
    acc_dt = 0.0
    for start in range(pid * BN, N, P * BN):
        offs_n = start + tl.arange(0, BN)
        mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)  # n*stride overflows int32 past N~8M
        xm = tl.load(XM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                     + offs_d[None, :] * sxd,
                     mask=mask[:, None], other=0.0).to(tl.float32)
        logits = _dot_w(xm, tl.trans(w_mat), DOT)
        logits = (logits + bias[None, :]) / tau
        m = tl.max(logits, axis=1)
        e = tl.exp(logits - m[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(mask[:, None], w, 0.0)

        dout = tl.load(DOUT + b * sob + h * soh + offs_n64[:, None] * son
                       + offs_d[None, :] * sod,
                       mask=mask[:, None], other=0.0).to(tl.float32)
        acc_dtok += _dot(tl.trans(w), dout, DOT)
        dw = _dot(dout, tl.trans(tok), DOT)
        gsum = tl.sum(dw * w, axis=1)
        dl = w * (dw - gsum[:, None])
        dlr = dl / tau
        dxm = _dot(dlr, w_mat, DOT)
        tl.store(DXM + b * sxb + h * sxh + offs_n64[:, None] * sxn
                 + offs_d[None, :] * sxd,
                 dxm.to(DXM.dtype.element_ty), mask=mask[:, None])
        acc_dw += _dot(tl.trans(dlr), xm, DOT)
        acc_db += tl.sum(dlr, axis=0)
        acc_dt += tl.sum(tl.sum(dl * (-logits / tau), axis=1), axis=0)

    idx = bh * P + pid
    tl.store(PDTOK + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dtok)
    tl.store(PDW + idx * G * D + offs_g[:, None] * D + offs_d[None, :], acc_dw)
    tl.store(PDB + idx * G + offs_g, acc_db)
    tl.store(PDT + idx, acc_dt)


# --------------------------------------------------------------------------- #
# custom ops (torch.library) — graph-capturable: torch.compile keeps a single
# graph and treats the kernels as opaque nodes. The earlier autograd.Function
# wrappers forced 33 Dynamo graph breaks per forward ("illegal getattr
# stride"), costing fused+compile the Inductor fusions that eager+compile
# enjoys (review P0-b, job 3084499).
# --------------------------------------------------------------------------- #

_LIB = torch.library.Library("cameleon", "DEF")
_LIB.define(
    "flash_slice(Tensor x_mid, Tensor fx_mid, Tensor weight, Tensor bias, "
    "Tensor tau, int dot) -> (Tensor, Tensor)")
_LIB.define(
    "flash_slice_bwd(Tensor x_mid, Tensor fx_mid, Tensor weight, Tensor bias, "
    "Tensor tau, Tensor dz_num, Tensor ds, int dot) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)")
_LIB.define(
    "flash_deslice(Tensor x_mid, Tensor weight, Tensor bias, Tensor tau, "
    "Tensor tokens, int dot) -> Tensor")
_LIB.define(
    "flash_deslice_bwd(Tensor x_mid, Tensor weight, Tensor bias, Tensor tau, "
    "Tensor tokens, Tensor d_out, int dot) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)")


def _slice_impl(x_mid, fx_mid, weight, bias, tau, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    bn, warps, stages = _cfg("slice_fwd", x_mid, dot, G, D)
    if dot == 1:  # only tf32 triggers the Triton 3.0 pipeliner segfault;
        stages = 1  # bf16 dots survive stages>1 (probe_bf16_stages.py)
    P = _n_programs(N, B * H, bn)
    part_z = torch.empty(B * H * P, G, D, device=x_mid.device,
                         dtype=torch.float32)
    part_s = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    _slice_fwd_kernel[(P, B * H)](
        x_mid, fx_mid, weight, bias, tau, part_z, part_s,
        N, P, H, *_strides(x_mid), *_strides(fx_mid),
        D=D, G=G, BN=bn, DOT=dot, num_warps=warps, num_stages=stages)
    return part_z.view(B, H, P, G, D).sum(2), part_s.view(B, H, P, G).sum(2)


def _slice_bwd_impl(x_mid, fx_mid, weight, bias, tau, dz_num, ds, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    dz_num = dz_num.contiguous().float()
    ds = ds.contiguous().float()
    dxm = torch.empty_like(x_mid)
    dfx = torch.empty_like(fx_mid)
    bn, warps, stages = _cfg("slice_bwd", x_mid, dot, G, D)
    if dot == 1:  # only tf32 triggers the Triton 3.0 pipeliner segfault;
        stages = 1  # bf16 dots survive stages>1 (probe_bf16_stages.py)
    P = _n_programs(N, B * H, bn)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    pdt = torch.empty(B * H * P, device=x_mid.device, dtype=torch.float32)
    _slice_bwd_kernel[(P, B * H)](
        x_mid, fx_mid, weight, bias, tau, dz_num, ds,
        dxm, dfx, pdw, pdb, pdt,
        N, P, H, *_strides(x_mid), *_strides(fx_mid),
        D=D, G=G, BN=bn, DOT=dot, num_warps=warps, num_stages=stages)
    return dxm, dfx, pdw.sum(0), pdb.sum(0), pdt.view(B, H, P).sum(dim=(0, 2))


def _deslice_impl(x_mid, weight, bias, tau, tokens, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    weight, bias, tau = weight.contiguous(), bias.contiguous(), tau.contiguous()
    tokens = tokens.contiguous()
    out = torch.empty(B, N, H, D, device=x_mid.device, dtype=x_mid.dtype)
    bn, warps, stages = _cfg("deslice_fwd", x_mid, dot, G, D)
    grid = (triton.cdiv(N, bn), B * H)
    _deslice_fwd_kernel[grid](
        x_mid, weight, bias, tau, tokens, out,
        N, H, *_strides(x_mid), *_strides(out),
        D=D, G=G, BN=bn, DOT=dot, num_warps=warps, num_stages=stages)
    return out


def _deslice_bwd_impl(x_mid, weight, bias, tau, tokens, d_out, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    tokens = tokens.contiguous()
    d_out = d_out.contiguous()
    dxm = torch.empty_like(x_mid)
    bn, warps, stages = _cfg("deslice_bwd", x_mid, dot, G, D)
    if dot == 1:  # only tf32 triggers the Triton 3.0 pipeliner segfault;
        stages = 1  # bf16 dots survive stages>1 (probe_bf16_stages.py)
    P = _n_programs(N, B * H, bn)
    pdtok = torch.empty(B * H * P, G, D, device=x_mid.device,
                        dtype=torch.float32)
    pdw = torch.empty(B * H * P, G, D, device=x_mid.device, dtype=torch.float32)
    pdb = torch.empty(B * H * P, G, device=x_mid.device, dtype=torch.float32)
    pdt = torch.empty(B * H * P, device=x_mid.device, dtype=torch.float32)
    _deslice_bwd_kernel[(P, B * H)](
        x_mid, weight, bias, tau, tokens, d_out,
        dxm, pdtok, pdw, pdb, pdt,
        N, P, H, *_strides(x_mid), *_strides(d_out),
        D=D, G=G, BN=bn, DOT=dot, num_warps=warps, num_stages=stages)
    return (dxm, pdtok.view(B, H, P, G, D).sum(2).to(tokens.dtype),
            pdw.sum(0), pdb.sum(0), pdt.view(B, H, P).sum(dim=(0, 2)))


_LIB.impl("flash_slice", _slice_impl, "CUDA")
_LIB.impl("flash_slice_bwd", _slice_bwd_impl, "CUDA")
_LIB.impl("flash_deslice", _deslice_impl, "CUDA")
_LIB.impl("flash_deslice_bwd", _deslice_bwd_impl, "CUDA")


@torch.library.register_fake("cameleon::flash_slice")
def _(x_mid, fx_mid, weight, bias, tau, dot):
    B, N, H, D = x_mid.shape
    G = weight.shape[0]
    return (x_mid.new_empty((B, H, G, D), dtype=torch.float32),
            x_mid.new_empty((B, H, G), dtype=torch.float32))


@torch.library.register_fake("cameleon::flash_slice_bwd")
def _(x_mid, fx_mid, weight, bias, tau, dz_num, ds, dot):
    return (torch.empty_like(x_mid), torch.empty_like(fx_mid),
            torch.empty_like(weight), torch.empty_like(bias),
            torch.empty_like(tau))


@torch.library.register_fake("cameleon::flash_deslice")
def _(x_mid, weight, bias, tau, tokens, dot):
    B, N, H, D = x_mid.shape
    return x_mid.new_empty((B, N, H, D))


@torch.library.register_fake("cameleon::flash_deslice_bwd")
def _(x_mid, weight, bias, tau, tokens, d_out, dot):
    return (torch.empty_like(x_mid), torch.empty_like(tokens),
            torch.empty_like(weight), torch.empty_like(bias),
            torch.empty_like(tau))


def _slice_setup(ctx, inputs, output):
    x_mid, fx_mid, weight, bias, tau, dot = inputs
    ctx.save_for_backward(x_mid, fx_mid, weight, bias, tau)
    ctx.dot = dot


def _slice_grad(ctx, dz_num, ds):
    x_mid, fx_mid, weight, bias, tau = ctx.saved_tensors
    if ds is None:
        ds = torch.zeros(dz_num.shape[:-1], device=dz_num.device,
                         dtype=torch.float32)
    dxm, dfx, dw, db, dtau = torch.ops.cameleon.flash_slice_bwd(
        x_mid, fx_mid, weight, bias, tau, dz_num, ds, ctx.dot)
    return dxm, dfx, dw, db, dtau, None


def _deslice_setup(ctx, inputs, output):
    x_mid, weight, bias, tau, tokens, dot = inputs
    ctx.save_for_backward(x_mid, weight, bias, tau, tokens)
    ctx.dot = dot


def _deslice_grad(ctx, d_out):
    x_mid, weight, bias, tau, tokens = ctx.saved_tensors
    dxm, dtok, dw, db, dtau = torch.ops.cameleon.flash_deslice_bwd(
        x_mid, weight, bias, tau, tokens, d_out, ctx.dot)
    return dxm, dw, db, dtau, dtok, None


torch.library.register_autograd("cameleon::flash_slice", _slice_grad,
                                setup_context=_slice_setup)
torch.library.register_autograd("cameleon::flash_deslice", _deslice_grad,
                                setup_context=_deslice_setup)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def fused_slice(x_mid, fx_mid, weight, bias, tau):
    """x_mid, fx_mid: (B, N, H, D); weight: (G, D); bias: (G,); tau: (H,).
    Returns fp32 z_num (B, H, G, D) and s (B, H, G); normalize outside as
    z = z_num / (s + eps)[..., None]."""
    _check_dims(x_mid.shape[3], weight.shape[0])
    return torch.ops.cameleon.flash_slice(x_mid, fx_mid, weight, bias, tau,
                                          _dot_mode(x_mid))


def fused_deslice(x_mid, weight, bias, tau, tokens):
    """tokens: (B, H, G, D) mixed tokens z'. Returns out (B, N, H, D) in
    x_mid's dtype — already in the layout to_out expects after a reshape."""
    _check_dims(x_mid.shape[3], weight.shape[0])
    return torch.ops.cameleon.flash_deslice(x_mid, weight, bias, tau, tokens,
                                            _dot_mode(x_mid))
