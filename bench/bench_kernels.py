"""Phase 1.5 micro-tuning sweep for the four fused_slice kernels.

Direct kernel launches (no autograd) at the paper head width (H=8, D=32),
sweeping BLOCK_N x num_warps x num_stages x dot precision (ieee/tf32/bf16)
per kernel, reporting time, achieved GB/s (logical N*H*D tensor traffic)
and TFLOP/s. The winners get hardcoded into fused_slice._CFG.

--slices selects G, the slice count: the kernels hold the whole G axis in
one tile, so the tile budget is BLOCK_N x G and the winner moves with G.
_CFG is keyed by G for exactly the supported set {16, 32, 64, 128}
(fused_slice._check_dims), one sweep job per value.
Run via kernel_bench.sbatch.
"""

import argparse
import itertools
import json
import os
import statistics

import torch
import triton

from flashslice.kernels import fused_slice as fs

# logical (N,H,D)-sized tensors touched / tl.dot calls, per kernel
_NBYTES = {"slice_fwd": 2, "deslice_fwd": 2, "slice_bwd": 4, "deslice_bwd": 3}
_NDOTS = {"slice_fwd": 2, "deslice_fwd": 2, "slice_bwd": 5, "deslice_bwd": 6}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="262144,1048576,4194304")
    p.add_argument("--dtypes", default="fp32,bf16")
    p.add_argument("--out",
                   default="scripts/kernel_study/results/phase15_kernels.json")
    p.add_argument("--defaults-only", action="store_true",
                   help="time only the production _CFG configs (P2 table)")
    p.add_argument("--slices", type=int, default=32, help="G (slice count)")
    p.add_argument("--warps", default="4,8", help="num_warps values to sweep")
    p.add_argument("--bns", default="64,128,256",
                   help="BLOCK_N values to sweep; drop the large ones at large "
                        "G, where BLOCK_N x G spills and a single timing run "
                        "costs minutes")
    p.add_argument("--verbose", action="store_true",
                   help="print every config before launching it (locates "
                        "uncatchable Triton aborts)")
    p.add_argument("--dim-head", type=int, default=32, help="D (head width)")
    a = p.parse_args()
    if a.defaults_only:
        a.ref_bw = copy_bw()
        print("D2D copy reference: {:.0f} GB/s (2x bytes / time; "
              "GH200 HBM3 spec ~4000 GB/s)".format(a.ref_bw), flush=True)
    B, H, D, G = 1, 8, a.dim_head, a.slices
    print("dims: B={} H={} D={} G={}".format(B, H, D, G), flush=True)
    results, best = {}, {}

    for n_s in a.ns.split(","):
        N = int(n_s)
        for dt_s in a.dtypes.split(","):
            dt = torch.float32 if dt_s == "fp32" else torch.bfloat16
            torch.manual_seed(0)
            xm = torch.randn(B, N, H, D, device="cuda", dtype=dt)
            fx = torch.randn(B, N, H, D, device="cuda", dtype=dt)
            W = torch.randn(G, D, device="cuda")
            bias = torch.randn(G, device="cuda")
            tau = torch.full((H,), 0.5, device="cuda")
            tok = torch.randn(B, H, G, D, device="cuda")
            dzn = torch.randn(B, H, G, D, device="cuda")
            ds = torch.randn(B, H, G, device="cuda")
            dout = torch.randn(B, N, H, D, device="cuda", dtype=dt)
            out = torch.empty_like(xm)
            dxm = torch.empty_like(xm)
            dfx = torch.empty_like(fx)
            sx, so = fs._strides(xm), fs._strides(out)
            key0 = "{}::{}".format(N, dt_s)
            results[key0] = {}

            def make_launches(bn, warps, stages, dot):
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
                        xm, fx, W, bias, tau, pz, ps, N, P, H, *sx, *sx, **kw),
                    "deslice_fwd": lambda: fs._deslice_fwd_kernel[
                        (triton.cdiv(N, bn), B * H)](
                        xm, W, bias, tau, tok, out, N, H, *sx, *so, **kw),
                    "slice_bwd": lambda: fs._slice_bwd_kernel[(P, B * H)](
                        xm, fx, W, bias, tau, dzn, ds, dxm, dfx,
                        pdw, pdb, pdt, N, P, H, *sx, *sx, **kw),
                    "deslice_bwd": lambda: fs._deslice_bwd_kernel[(P, B * H)](
                        xm, W, bias, tau, tok, dout, dxm,
                        pdtok, pdw, pdb, pdt, N, P, H, *sx, *so, **kw),
                }

            if a.defaults_only:
                # P2 table: production _CFG configs only, every dot mode valid
                # for the dtype, with theoretical-minimum bytes and % of the
                # measured copy peak.
                dlbl = {0: "ieee", 1: "tf32", 2: "bf16v", 3: "bf16"}
                dots = (0, 1) if dt == torch.float32 else (0, 1, 2, 3)
                for kname in ("slice_fwd", "deslice_fwd", "slice_bwd",
                              "deslice_bwd"):
                    for dot in dots:
                        bn, warps, stages = fs._cfg(kname, xm, dot, G, D)
                        st = 1 if (dot == 1 and kname != "deslice_fwd") \
                            else stages
                        ms = timeit(make_launches(bn, warps, st, dot)[kname])
                        gb = (_NBYTES[kname] * N * H * D * xm.element_size()
                              / 1e9)
                        gbps = gb * 1e3 / ms
                        tag = "{}/{}".format(kname, dlbl[dot])
                        results[key0][tag] = {
                            "ms": ms, "GB": gb, "GBps": gbps,
                            "pct_copy_peak": 100 * gbps / a.ref_bw,
                            "TFLOPs": _NDOTS[kname] * 2 * N * H * G * D
                                      / ms / 1e9,
                        }
                        print("  {:12s} {:4s} N={:>8} {:5s}: {:7.3f} ms  "
                              "{:6.3f} GB  {:7.0f} GB/s  {:5.1f}% of copy peak"
                              .format(kname, dt_s, N, dlbl[dot],
                                      ms, gb, gbps, 100 * gbps / a.ref_bw),
                              flush=True)
                continue

            for bn, warps, stages, dot in itertools.product(
                    [int(v) for v in a.bns.split(",")],
                    [int(w) for w in a.warps.split(",")],
                    (1, 2, 3), (0, 1, 3)):
                if dot == 3 and dt == torch.float32:
                    continue  # bf16 dots are for 16-bit inputs only
                launches = make_launches(bn, warps, stages, dot)
                for kname, fn in launches.items():
                    if dot == 1 and stages != 1 and kname != "deslice_fwd":
                        # Triton 3.0's loop pipeliner segfaults compiling async
                        # tf32 dots (uncatchable); bf16 dots survive.
                        continue
                    tag = "{}/bn{}w{}s{}/{}".format(
                        kname, bn, warps, stages,
                        {0: "ieee", 1: "tf32", 3: "bf16"}[dot])
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
                        "GBps": _NBYTES[kname] * N * H * D * xm.element_size()
                                / ms / 1e6,
                        "TFLOPs": _NDOTS[kname] * 2 * N * H * G * D / ms / 1e9,
                    }
                    k = (kname, dt_s, N, dot)
                    if k not in best or ms < best[k][0]:
                        best[k] = (ms, bn, warps, stages)
            print("swept {}".format(key0), flush=True)

    print("\n===== best per (kernel, dtype, N, precision) =====")
    for (kname, dt_s, N, dot), (ms, bn, warps, stages) in sorted(
            best.items(), key=str):
        print("{:12s} {:4s} N={:>8} {}: {:7.3f} ms  bn={:>3} warps={} stages={}"
              .format(kname, dt_s, N, {0: "ieee", 1: "tf32", 3: "bf16"}[dot],
                      ms, bn, warps, stages), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"dims": {"B": B, "H": H, "D": D, "G": G},
                   "results": results,
                   "best": {str(k): v for k, v in best.items()}}, f, indent=1)
    print("wrote {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
