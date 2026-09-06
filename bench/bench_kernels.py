"""Tile-tuning sweep for the fused slice/deslice kernels.

Direct kernel launches (no autograd), sweeping BLOCK_N x num_warps x
num_stages x dot precision per kernel and reporting time, achieved GB/s
(logical N*H*D tensor traffic) and TFLOP/s. The winners get hardcoded into
slice_ops._CFG (single-tile family) or blocked._CFG_BLK (blocked family).

--slices selects G, the slice count; --dim-head the logits width D of x_mid
and the weight; --dim-value the value width DV of fx_mid / tokens / out
(default D; a different value routes to the blocked family, where the tile
table may carry an entry per value width); --weight-shape whether the weight
is shared (G, D), per head (H, G, D) or per sample and head (B, H, G, D) --
read through strides, so it should not move the numbers, and this is how to
check that.

The single-tile kernels hold the whole G axis in one tile, so the tile
budget is BLOCK_N x G and the winner moves with G. _CFG is keyed by G for
exactly the single-tile set {16, 32, 64, 128} (slice_ops.single_tile_dims),
one sweep job per value.

--family blocked sweeps the G-blocked kernels of kernels/blocked.py at the
G given by --slices (any value; the block size comes from blocked.tiles, or
--block-g). The statistics kernel timed is the one the current stats mode
uses (blocked.set_stats_mode / FLASHSLICE_STATS_MODE); deslice_fwd_online
is the deslice that forms its own statistics. Where _CFG_BLK has no entry
the blocked kernels borrow the single-tile tables keyed by their block size
(blocked._FAMILY), so this is how to check that borrowing and find their
own winners; bench/pick_tiles.py turns the JSON into a table entry. For
layer-level numbers at any shape see bench_layer.py. GPU required.
"""

import argparse
import itertools
import json
import os
import statistics

import torch
import triton

from flashslice.kernels import blocked as fb
from flashslice.kernels import slice_ops as fs

# logical (N,H,D)-sized tensors touched / tl.dot calls, per kernel
_NBYTES = {"slice_fwd": 2, "deslice_fwd": 2, "slice_bwd": 4, "deslice_bwd": 3}
_NDOTS = {"slice_fwd": 2, "deslice_fwd": 2, "slice_bwd": 5, "deslice_bwd": 6}
# the G-blocked family: (logits-width dots, value-width dots) per full G
# (each G-block contributes GB/G of one), so TFLOP/s stay comparable with
# the single-tile numbers. The two-pass statistics kernel does 2 logits dots.
_NBYTES_BLK = {"stats": 1, "slice_fwd_g": 2, "deslice_fwd_n": 2,
               "deslice_fwd_online": 2, "slice_bwd_n": 4, "slice_bwd_g": 2,
               "deslice_bwd_n": 3, "deslice_bwd_g": 2}
_NDOTS_BLK = {"stats": (1, 0), "slice_fwd_g": (1, 1), "deslice_fwd_n": (1, 1),
              "deslice_fwd_online": (1, 1), "slice_bwd_n": (3, 3),
              "slice_bwd_g": (2, 1), "deslice_bwd_n": (3, 2),
              "deslice_bwd_g": (2, 2)}
_DOT_NAME = {0: "ieee", 1: "tf32", 2: "bf16v", 3: "bf16", 4: "tf32x3"}


def copy_bw():
    """Measured D2D copy bandwidth (read+write), the practical HBM reference."""
    src = torch.empty(512 * 1024 * 1024, device="cuda")  # 2 GiB fp32
    dst = torch.empty_like(src)
    ms = timeit(lambda: dst.copy_(src), iters=20, warmup=5)
    return 2 * src.numel() * 4 / ms / 1e6  # GB/s


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def make_weight(shape, B, H, G, D):
    """The slice weight and bias in the requested layout, contiguous."""
    if shape == "gd":
        return (torch.randn(G, D, device="cuda"), torch.randn(G, device="cuda"))
    if shape == "hgd":
        return (torch.randn(H, G, D, device="cuda"),
                torch.randn(H, G, device="cuda"))
    return (torch.randn(B, H, G, D, device="cuda"),
            torch.randn(B, H, G, device="cuda"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="262144,1048576,4194304")
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--out",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "results", "kernels.json"))
    p.add_argument("--defaults-only", action="store_true",
                   help="time only the production tile configs, per kernel")
    p.add_argument("--slices", type=int, default=32, help="G (slice count)")
    p.add_argument("--warps", default="4,8", help="num_warps values to sweep")
    p.add_argument("--bns", default="64,128,256",
                   help="BLOCK_N values to sweep; drop the large ones at large "
                        "G, where BLOCK_N x G spills and a single timing run "
                        "costs minutes")
    p.add_argument("--stages", default="1,2,3", help="num_stages values to sweep")
    p.add_argument("--verbose", action="store_true",
                   help="print every config before launching it (locates "
                        "uncatchable Triton aborts)")
    p.add_argument("--dim-head", type=int, default=32,
                   help="D, the logits width of x_mid and the weight")
    p.add_argument("--dim-value", type=int, default=None,
                   help="DV, the value width of fx_mid / tokens / out "
                        "(default: D; DV != D takes the blocked family)")
    p.add_argument("--weight-shape", choices=("gd", "hgd", "bhgd"), default="gd",
                   help="slice weight shared (G, D), per head (H, G, D) or "
                        "per sample and head (B, H, G, D)")
    p.add_argument("--family", choices=("single", "blocked"), default="single",
                   help="single-tile kernels (slice_ops) or the G-blocked "
                        "ones (blocked)")
    p.add_argument("--block-g", type=int, default=None,
                   help="G-block size for --family blocked (default: "
                        "blocked.tiles)")
    p.add_argument("--dots", default="ieee,tf32,bf16",
                   help="dot modes to sweep (ieee, tf32, bf16, tf32x3)")
    a = p.parse_args()
    B, H, D, G = 1, 8, a.dim_head, a.slices
    DV = a.dim_value or D
    if a.family == "single" and DV != D:
        raise SystemExit("the single-tile family needs --dim-value == --dim-head")
    DT = GB = DVT = None
    if a.family == "blocked":
        fb.set_block_g(a.block_g)
        DT, GB = fb.tiles(D, G)
        DVT = fb._pow2_at_least_16(DV)
        NGB = triton.cdiv(G, GB)
        print("blocked family: D_tile={} G_block={} DV_tile={} ({} block(s)), "
              "stats mode {}".format(DT, GB, DVT, NGB, fb.stats_mode()), flush=True)
    nbytes = _NBYTES_BLK if a.family == "blocked" else _NBYTES
    if a.defaults_only:
        a.ref_bw = copy_bw()
        print("D2D copy reference: {:.0f} GB/s (2x bytes / time; "
              "GH200 HBM3 spec ~4000 GB/s)".format(a.ref_bw), flush=True)
    print("dims: B={} H={} D={} DV={} G={} weight {}".format(
        B, H, D, DV, G, a.weight_shape), flush=True)
    results, best = {}, {}

    def flops(kname, N):
        if a.family == "blocked":
            nd, nv = _NDOTS_BLK[kname]
            if kname == "stats" and fb.stats_mode() == "two-pass":
                nd = 2
            return 2 * N * H * G * (nd * D + nv * DV)
        return _NDOTS[kname] * 2 * N * H * G * D

    for n_s in a.ns.split(","):
        N = int(n_s)
        for dt_s in a.dtypes.split(","):
            dt = torch.float32 if dt_s == "fp32" else torch.bfloat16
            torch.manual_seed(0)
            xm = torch.randn(B, N, H, D, device="cuda", dtype=dt)
            fx = torch.randn(B, N, H, DV, device="cuda", dtype=dt)
            W, bias = make_weight(a.weight_shape, B, H, G, D)
            wb = fs._wb_layout(W, bias, B, H)
            tau = torch.full((H,), 0.5, device="cuda")
            tok = torch.randn(B, H, G, DV, device="cuda")
            dzn = torch.randn(B, H, G, DV, device="cuda")
            ds = torch.randn(B, H, G, device="cuda")
            dout = torch.randn(B, N, H, DV, device="cuda", dtype=dt)
            out = torch.empty_like(fx)
            dxm = torch.empty_like(xm)
            dfx = torch.empty_like(fx)
            sx, sf, so = fs._strides(xm), fs._strides(fx), fs._strides(out)
            key0 = "{}::{}".format(N, dt_s)
            results[key0] = {}

            if a.family == "blocked":
                # statistics and delta as the production path would produce
                # them, so every kernel times against valid inputs
                stats = fb.compute_stats(xm, W, bias, tau)
                stats2 = torch.empty_like(stats)
                delta = torch.empty(B, H, N, device="cuda")

            def make_launches_blocked(bn, warps, stages, dot):
                P = fs._n_programs(N, B * H * NGB, bn)
                pz = torch.empty(B * H * P, G, DV, device="cuda")
                ps = torch.empty(B * H * P, G, device="cuda")
                pdw = torch.empty(B * H * P, G, D, device="cuda")
                pdb = torch.empty(B * H * P, G, device="cuda")
                pdt = torch.empty(B * H * triton.cdiv(N, bn), device="cuda")
                pdtok = torch.empty(B * H * P, G, DV, device="cuda")
                kw = fb._consts(D, G, DT, GB, bn, dot, warps, stages, DV)
                gn = (triton.cdiv(N, bn), B * H)
                gg = (NGB, P, B * H)
                stats_kernel = (fb._stats_kernel if fb.stats_mode() == "online"
                                else fb._stats_twopass_kernel)
                return {
                    "stats": lambda: stats_kernel[gn](
                        xm, W, bias, tau, stats2, N, G, H, *wb, *sx, **kw),
                    "slice_fwd_g": lambda: fb._slice_fwd_g_kernel[gg](
                        xm, fx, W, bias, tau, stats, pz, ps, N, G, P, H,
                        *wb, *sx, *sf, **kw),
                    "deslice_fwd_n": lambda: fb._deslice_fwd_n_kernel[gn](
                        xm, W, bias, tau, tok, stats, out, N, G, H,
                        *wb, *sx, *so, **kw),
                    "deslice_fwd_online": lambda: fb._deslice_fwd_online_kernel[gn](
                        xm, W, bias, tau, tok, stats2, out, N, G, H,
                        *wb, *sx, *so, **kw),
                    "slice_bwd_n": lambda: fb._slice_bwd_n_kernel[gn](
                        xm, fx, W, bias, tau, stats, dzn, ds, dxm, dfx, delta,
                        pdt, N, G, H, *wb, *sx, *sf, **kw),
                    "slice_bwd_g": lambda: fb._slice_bwd_g_kernel[gg](
                        xm, fx, W, bias, tau, stats, delta, dzn, ds,
                        pdw, pdb, N, G, P, H, *wb, *sx, *sf, **kw),
                    "deslice_bwd_n": lambda: fb._deslice_bwd_n_kernel[gn](
                        xm, W, bias, tau, tok, stats, dout, dxm, delta, pdt,
                        N, G, H, *wb, *sx, *so, **kw),
                    "deslice_bwd_g": lambda: fb._deslice_bwd_g_kernel[gg](
                        xm, W, bias, tau, tok, stats, delta, dout,
                        pdtok, pdw, pdb, N, G, P, H, *wb, *sx, *so, **kw),
                }

            def make_launches_single(bn, warps, stages, dot):
                P = fs._n_programs(N, B * H, bn)
                pz = torch.empty(B * H * P, G, D, device="cuda")
                ps = torch.empty(B * H * P, G, device="cuda")
                pdw = torch.empty(B * H * P, G, D, device="cuda")
                pdb = torch.empty(B * H * P, G, device="cuda")
                pdt = torch.empty(B * H * P, device="cuda")
                pdtok = torch.empty(B * H * P, G, D, device="cuda")
                kw = dict(D=D, G=G, BN=bn, DOT=dot,
                          num_warps=warps, num_stages=stages)
                return {
                    "slice_fwd": lambda: fs._slice_fwd_kernel[(P, B * H)](
                        xm, fx, W, bias, tau, pz, ps, N, P, H, *wb, *sx, *sf,
                        **kw),
                    "deslice_fwd": lambda: fs._deslice_fwd_kernel[
                        (triton.cdiv(N, bn), B * H)](
                        xm, W, bias, tau, tok, out, N, H, *wb, *sx, *so, **kw),
                    "slice_bwd": lambda: fs._slice_bwd_kernel[(P, B * H)](
                        xm, fx, W, bias, tau, dzn, ds, dxm, dfx,
                        pdw, pdb, pdt, N, P, H, *wb, *sx, *sf, **kw),
                    "deslice_bwd": lambda: fs._deslice_bwd_kernel[(P, B * H)](
                        xm, W, bias, tau, tok, dout, dxm,
                        pdtok, pdw, pdb, pdt, N, P, H, *wb, *sx, *so, **kw),
                }

            make_launches = (make_launches_blocked if a.family == "blocked"
                             else make_launches_single)

            def default_cfg(kname, dot):
                if a.family == "blocked":
                    return fb._launch_cfg(kname, xm, dot, GB, DT, DVT)
                bn, warps, stages = fs._cfg(kname, xm, dot, G, D)
                if kname != "deslice_fwd":
                    stages = fs._stages(dot, stages)
                return bn, warps, stages

            if a.defaults_only:
                # production tile configs only, every dot mode valid for the
                # dtype, with theoretical-minimum bytes and % of the measured
                # copy peak.
                dots = (0, 1, 4) if dt == torch.float32 else (0, 1, 2, 3, 4)
                for kname in nbytes:
                    for dot in dots:
                        bn, warps, st = default_cfg(kname, dot)
                        ms = timeit(make_launches(bn, warps, st, dot)[kname])
                        gb = (nbytes[kname] * N * H * D * xm.element_size()
                              / 1e9)
                        gbps = gb * 1e3 / ms
                        tag = "{}/{}".format(kname, _DOT_NAME[dot])
                        results[key0][tag] = {
                            "ms": ms, "GB": gb, "GBps": gbps,
                            "pct_copy_peak": 100 * gbps / a.ref_bw,
                            "TFLOPs": flops(kname, N) / ms / 1e9,
                            "cfg": [bn, warps, st],
                        }
                        print("  {:18s} {:4s} N={:>8} {:6s}: {:7.3f} ms  "
                              "{:6.3f} GB  {:7.0f} GB/s  {:5.1f}% of copy peak"
                              "  {:6.1f} TFLOP/s  bn{} w{} s{}"
                              .format(kname, dt_s, N, _DOT_NAME[dot],
                                      ms, gb, gbps, 100 * gbps / a.ref_bw,
                                      results[key0][tag]["TFLOPs"],
                                      bn, warps, st),
                              flush=True)
                continue

            dot_levels = [{"ieee": 0, "tf32": 1, "bf16": 3, "tf32x3": 4}[d]
                          for d in a.dots.split(",")]
            for bn, warps, stages, dot in itertools.product(
                    [int(v) for v in a.bns.split(",")],
                    [int(w) for w in a.warps.split(",")],
                    [int(s) for s in a.stages.split(",")], dot_levels):
                if dot == 3 and dt == torch.float32:
                    continue  # bf16 dots are for 16-bit inputs only
                launches = make_launches(bn, warps, stages, dot)
                for kname, fn in launches.items():
                    if dot in (1, 4) and stages != 1 and kname != "deslice_fwd":
                        # Triton 3.0's loop pipeliner segfaults compiling async
                        # tf32 dots (uncatchable); bf16 dots survive.
                        continue
                    if dot and warps > 4 and ("bwd" in kname
                                              or kname == "deslice_fwd_online"):
                        # Triton 3.0 aborts (assert) on an mma -> mma layout
                        # conversion in backward kernels and in the online
                        # deslice with tensor-core dots on 8 warps
                        # (blocked._launch_cfg, slice_ops._cfg).
                        continue
                    tag = "{}/bn{}w{}s{}/{}".format(
                        kname, bn, warps, stages, _DOT_NAME[dot])
                    if a.verbose:
                        # Triton aborts (not raises) on some shapes — e.g. the
                        # MMA-N assert at G=16 — so the last line printed is
                        # the culprit.
                        print("  try {} {}".format(key0, tag), flush=True)
                    try:
                        ms = timeit(fn)
                    except Exception as err:  # noqa: BLE001 - config infeasible
                        results[key0][tag] = "ERR " + str(err)[:60]
                        continue
                    results[key0][tag] = {
                        "ms": ms,
                        "GBps": nbytes[kname] * N * H * D * xm.element_size()
                                / ms / 1e6,
                        "TFLOPs": flops(kname, N) / ms / 1e9,
                    }
                    k = (kname, dt_s, N, dot)
                    if k not in best or ms < best[k][0]:
                        best[k] = (ms, bn, warps, stages)
                _dump(a, B, H, D, DV, G, results, best)  # survive a later abort
            print("swept {}".format(key0), flush=True)

    print("\n===== best per (kernel, dtype, N, precision) =====")
    for (kname, dt_s, N, dot), (ms, bn, warps, stages) in sorted(
            best.items(), key=str):
        print("{:18s} {:4s} N={:>8} {}: {:7.3f} ms  bn={:>3} warps={} stages={}"
              .format(kname, dt_s, N, _DOT_NAME[dot], ms, bn, warps, stages),
              flush=True)

    _dump(a, B, H, D, DV, G, results, best)
    print("wrote {}".format(a.out), flush=True)


def _dump(a, B, H, D, DV, G, results, best):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dims = {"B": B, "H": H, "D": D, "DV": DV, "G": G, "family": a.family,
            "weight_shape": a.weight_shape}
    if a.family == "blocked":
        dims["D_tile"], dims["G_block"] = fb.tiles(a.dim_head, a.slices)
        dims["DV_tile"] = fb._pow2_at_least_16(DV)
        dims["stats_mode"] = fb.stats_mode()
    with open(a.out, "w") as f:
        json.dump({"dims": dims, "results": results,
                   "best": {str(k): v for k, v in best.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
