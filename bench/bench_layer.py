"""Layer-level time and memory, eager vs fused, at any shape.

One Physics_Attention_Irregular_Mesh, forward or training step (fwd + bwd),
eager and fused inside the same process, medians of --iters iterations and
the peak allocated memory of one iteration. This is the number a user feels;
bench_kernels.py is the per-kernel view for tuning.

Shapes outside the single-tile range (G a power of two in [16, 128], same for
D) run the G-blocked kernels automatically; --kernels forces a path so the
two can be compared on a shape both serve.

    python bench_layer.py --slices 256 --ns 262144,1048576 --dtype bf16
    python bench_layer.py --kernels blocked --slices 32      # blocked on the paper shape

GPU required. Run one benchmark job at a time on the GPU.
"""

import argparse
import statistics

import torch

from flashslice.kernels import set_kernel_mode, single_tile_dims
from flashslice.layers import Physics_Attention_Irregular_Mesh


def _sync_time(fn, iters, warmup):
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


def measure(layer, x, mode, dtype, iters, warmup):
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16")

    def fwd():
        with torch.no_grad(), autocast:
            layer(x)[0]

    def step():
        with autocast:
            out, _ = layer(x)
        out.float().square().mean().backward()
        layer.zero_grad(set_to_none=True)

    fn = fwd if mode == "fwd" else step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    return _sync_time(fn, iters, warmup), peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", default="262144")
    p.add_argument("--slices", type=int, default=32, help="G")
    p.add_argument("--dim-head", type=int, default=32, help="D")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--mode", choices=("fwd", "train"), default="train")
    p.add_argument("--kernels", choices=("auto", "single-tile", "blocked"),
                   default="auto")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--skip-eager", action="store_true")
    a = p.parse_args()

    set_kernel_mode(a.kernels)
    H, D, G = a.heads, a.dim_head, a.slices
    path = a.kernels if a.kernels != "auto" else (
        "single-tile" if single_tile_dims(D, G) else "blocked")
    print("torch {}  {}  H={} D={} G={} dtype={} mode={} fused-kernels={}".format(
        torch.__version__, torch.cuda.get_device_name(0), H, D, G, a.dtype,
        a.mode, path), flush=True)
    print("{:>9} {:>10} {:>10} {:>8} {:>10} {:>10} {:>7}".format(
        "N", "eager ms", "fused ms", "speedup", "eager GB", "fused GB", "saved"),
        flush=True)
    for n_s in a.ns.split(","):
        N = int(n_s)
        torch.manual_seed(0)
        layer = Physics_Attention_Irregular_Mesh(
            H * D, heads=H, dim_head=D, slice_num=G, use_fused_slice=True).cuda()
        assert layer.use_fused_slice, layer.fused_slice_fallback
        x = torch.randn(1, N, H * D, device="cuda")
        res = {}
        for fused in ((False, True) if not a.skip_eager else (True,)):
            layer.use_fused_slice = fused
            try:
                res[fused] = measure(layer, x, a.mode, a.dtype, a.iters, a.warmup)
            except torch.cuda.OutOfMemoryError:
                res[fused] = None
            torch.cuda.empty_cache()
        e, f = res.get(False), res.get(True)
        fmt = lambda v, u: ("{:10.1f}".format(v) if v is not None else "{:>10}".format("OOM"))  # noqa: E731
        speed = "{:7.2f}x".format(e[0] / f[0]) if e and f else "{:>8}".format("-")
        saved = "{:6.0f}%".format(100 * (1 - f[1] / e[1])) if e and f else "{:>7}".format("-")
        print("{:>9} {} {} {} {} {} {}".format(
            N, fmt(e and e[0], "ms"), fmt(f and f[0], "ms"), speed,
            fmt(e and e[1], "GB"), fmt(f and f[1], "GB"), saved), flush=True)
        del layer, x


if __name__ == "__main__":
    main()
