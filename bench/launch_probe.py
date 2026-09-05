"""Launch the blocked kernels one at a time, each in its own process.

Triton 3.0 can abort the process (an assert, not an exception) on some tile
and warp combinations, and a kernel that touches memory it should not leaves
a sticky CUDA error that surfaces at the *next* launch or module load, under
another kernel's name. This probe launches every blocked kernel at a given
shape and configuration in a fresh Python process with CUDA_LAUNCH_BLOCKING=1,
synchronizes, and reports per kernel, so the culprit is named and the rest
still runs.

    python bench/launch_probe.py --slices 256 --dim-head 32 --dtype bf16 --dot bf16
    python bench/launch_probe.py ... --stages 1,2,3          # sweep a launch parameter
    python bench/launch_probe.py ... --kernel slice_bwd_n     # one kernel only

Configurations default to the production table for the shape (what
bench_kernels.py --defaults-only would time); --bn / --warps / --stages
override them for every kernel. GPU required.
"""

import argparse
import itertools
import os
import subprocess
import sys

KERNELS = ("stats", "slice_fwd_g", "deslice_fwd_n", "deslice_fwd_online",
           "slice_bwd_n", "slice_bwd_g", "deslice_bwd_n", "deslice_bwd_g")
DOT = {"ieee": 0, "tf32": 1, "bf16v": 2, "bf16": 3, "tf32x3": 4}


def _child(a):
    """Runs in the child process: one kernel, one configuration, one launch."""
    import torch
    import triton

    from flashslice.kernels import blocked as fb
    from flashslice.kernels import slice_ops as fs

    B, H, D, G = 1, 8, a.dim_head, a.slices
    DV = a.dim_value or D
    N = a.n
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    dot = DOT[a.dot]
    fb.set_block_g(a.block_g)
    DT, GB = fb.tiles(D, G)
    DVT = fb._pow2_at_least_16(DV)
    NGB = triton.cdiv(G, GB)
    torch.manual_seed(0)
    xm = torch.randn(B, N, H, D, device="cuda", dtype=dt)
    fx = torch.randn(B, N, H, DV, device="cuda", dtype=dt)
    W = torch.randn(G, D, device="cuda")
    bias = torch.randn(G, device="cuda")
    wb = fs._wb_layout(W, bias, B, H)
    tau = torch.full((H,), 0.5, device="cuda")
    tok = torch.randn(B, H, G, DV, device="cuda")
    dzn = torch.randn(B, H, G, DV, device="cuda")
    ds = torch.randn(B, H, G, device="cuda")
    dout = torch.randn(B, N, H, DV, device="cuda", dtype=dt)
    out = torch.empty_like(fx)
    dxm = torch.empty_like(xm)
    dfx = torch.empty_like(fx)
    sx, so = fs._strides(xm), fs._strides(out)
    stats = fb.compute_stats(xm, W, bias, tau)
    torch.cuda.synchronize()
    stats2 = torch.empty_like(stats)
    delta = torch.zeros(B, H, N, device="cuda")

    bn, warps, stages = fb._launch_cfg(a.kernel, xm, dot, GB, DT, DVT)
    bn = a.bn or bn
    warps = a.warps or warps
    stages = a.stages or stages
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
    own = dict(OWN_L=fb._OWN_L)
    stats_kernel = (fb._stats_kernel if fb.stats_mode() == "online"
                    else fb._stats_twopass_kernel)
    launches = {
        "stats": lambda: stats_kernel[gn](
            xm, W, bias, tau, stats2, N, G, H, *wb, *sx, **kw),
        "slice_fwd_g": lambda: fb._slice_fwd_g_kernel[gg](
            xm, fx, W, bias, tau, stats, pz, ps, N, G, P, H, *wb, *sx, *sx, **kw),
        "deslice_fwd_n": lambda: fb._deslice_fwd_n_kernel[gn](
            xm, W, bias, tau, tok, stats, out, N, G, H, *wb, *sx, *so, **kw),
        "deslice_fwd_online": lambda: fb._deslice_fwd_online_kernel[gn](
            xm, W, bias, tau, tok, stats2, out, N, G, H, *wb, *sx, *so, **kw),
        "slice_bwd_n": lambda: fb._slice_bwd_n_kernel[gn](
            xm, fx, W, bias, tau, stats, dzn, ds, dxm, dfx, delta, pdt,
            N, G, H, *wb, *sx, *sx, **own, **kw),
        "slice_bwd_g": lambda: fb._slice_bwd_g_kernel[gg](
            xm, fx, W, bias, tau, stats, delta, dzn, ds, pdw, pdb,
            N, G, P, H, *wb, *sx, *sx, **kw),
        "deslice_bwd_n": lambda: fb._deslice_bwd_n_kernel[gn](
            xm, W, bias, tau, tok, stats, dout, dxm, delta, pdt,
            N, G, H, *wb, *sx, *so, **own, **kw),
        "deslice_bwd_g": lambda: fb._deslice_bwd_g_kernel[gg](
            xm, W, bias, tau, tok, stats, delta, dout, pdtok, pdw, pdb,
            N, G, P, H, *wb, *sx, *so, **kw),
    }
    launches[a.kernel]()
    torch.cuda.synchronize()
    launches[a.kernel]()
    torch.cuda.synchronize()
    print("OK {} bn{} w{} s{} (D_tile={} G_block={} DV_tile={})".format(
        a.kernel, bn, warps, stages, DT, GB, DVT), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slices", type=int, default=256)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument("--dim-value", type=int, default=None)
    p.add_argument("--n", type=int, default=30013)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--dot", choices=tuple(DOT), default="bf16")
    p.add_argument("--block-g", type=int, default=None)
    p.add_argument("--kernel", default=None, help="one kernel (default: all)")
    p.add_argument("--bn", default=None, help="BLOCK_N value(s), comma separated")
    p.add_argument("--warps", default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a._child:
        a.bn = int(a.bn) if a.bn else None
        a.warps = int(a.warps) if a.warps else None
        a.stages = int(a.stages) if a.stages else None
        _child(a)
        return
    kernels = (a.kernel,) if a.kernel else KERNELS
    grid = itertools.product(
        kernels,
        (a.bn or "").split(",") if a.bn else [None],
        (a.warps or "").split(",") if a.warps else [None],
        (a.stages or "").split(",") if a.stages else [None])
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    failed = 0
    for kname, bn, warps, stages in grid:
        cmd = [sys.executable, os.path.abspath(__file__), "--_child", "--kernel", kname,
               "--slices", str(a.slices), "--dim-head", str(a.dim_head), "--n", str(a.n),
               "--dtype", a.dtype, "--dot", a.dot]
        if a.dim_value:
            cmd += ["--dim-value", str(a.dim_value)]
        if a.block_g:
            cmd += ["--block-g", str(a.block_g)]
        for flag, val in (("--bn", bn), ("--warps", warps), ("--stages", stages)):
            if val:
                cmd += [flag, val]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode == 0 and tail.startswith("OK"):
            print("  " + tail, flush=True)
        else:
            failed += 1
            err = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1][:200]
            print("  FAIL {} bn={} w={} s={} rc={}: {}".format(
                kname, bn, warps, stages, r.returncode, err), flush=True)
    print("{} failure(s)".format(failed) if failed else "all launches OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
