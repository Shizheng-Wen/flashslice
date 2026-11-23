# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""Replay the layer's own backward call at every tile of one blocked kernel.

bench_kernels.py times each kernel on inputs it builds itself. The layer can
hand a kernel other dtypes -- under bf16 autocast the deslice backward gets
bf16 tokens, the sweep gives it fp32 ones -- and Triton compiles the two
apart, so a sweep winner can be slow inside a training step. This script
takes the arguments of a real step (one layer, N = 262144, the blocked
kernels forced) and times that call with the table entry of one kernel set
to each (BLOCK_N, num_warps, num_stages). The time is the whole backward
call, both of its kernels.

    python bench/replay_tiles.py 256 bf16 deslice_bwd_n          # G, dtype, kernel
    python bench/replay_tiles.py 256 bf16 slice_bwd_g bf16       # ... and a dot mode

Kernels: slice_bwd_n, slice_bwd_g, deslice_bwd_n, deslice_bwd_g. GPU required.
"""
import itertools, statistics, sys, torch
import flashslice.kernels.blocked as fb
from flashslice.layers import Physics_Attention_Irregular_Mesh
G, dtype, kernel = int(sys.argv[1]), sys.argv[2], sys.argv[3]
DOT = sys.argv[4] if len(sys.argv) > 4 else ""
from flashslice.kernels import set_dot_mode; set_dot_mode(DOT)
N = 262144
impl = "_deslice_blk_bwd_impl" if kernel.startswith("deslice") else "_slice_blk_bwd_impl"
torch.manual_seed(0)
layer = Physics_Attention_Irregular_Mesh(256, heads=8, dim_head=32, slice_num=G, use_fused_slice=True).cuda()
from flashslice.kernels import set_kernel_mode; set_kernel_mode("blocked")
x = torch.randn(1, N, 256, device="cuda")
cap = {}; orig = getattr(fb, impl)
# The op calls its implementation by reference, so the arguments are taken at the kernel launch instead:
# the first 7 (deslice) or 8 (slice) positional arguments of the *_bwd_n kernel are the implementation's.
kname = "_deslice_bwd_n_kernel" if kernel.startswith("deslice") else "_slice_bwd_n_kernel"
narg = 7 if kernel.startswith("deslice") else 8
real = getattr(fb, kname)
class Spy:
    def __getitem__(self, grid):
        def run(*args, **kw):
            cap["a"] = tuple(args[:narg]) + (kw["DOT"],)
            return real[grid](*args, **kw)
        return run
setattr(fb, kname, Spy())
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
    out, _ = layer(x)
out.float().square().mean().backward()
setattr(fb, kname, real)
a = cap["a"]
print("args:", [(tuple(t.shape), str(t.dtype).replace("torch.", ""), t.is_contiguous()) if torch.is_tensor(t) else t for t in a])
x_mid, dot = a[0], a[-1]
D, Gw = x_mid.shape[3], G
dt, gb0 = fb.tiles(D, Gw, dot)
is16 = x_mid.dtype != torch.float32
key = (gb0, dt)
# the table in use on this GPU and this Triton
table = (fb._CFG_BLK_ADA if fb.tile_table() == "ada"
         else fb._CFG_BLK_T36 if fb.tile_series() == "3.6" else fb._CFG_BLK)
entry = table[key][kernel]
print("table key", key, "entry", (is16, dot), "current", entry.get((is16, dot)))
def t_ms():
    for _ in range(3): orig(*a)
    torch.cuda.synchronize(); ts = []
    for _ in range(7):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); orig(*a); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)
cur = entry.get((is16, dot)); base = t_ms(); print("current tile: %.2f ms (the whole backward call, both kernels)" % base)
res = []
for bn, w, s in itertools.product((32, 64, 128, 256), (4, 8), (1, 2, 3)):
    entry[(is16, dot)] = (bn, w, s)
    try: res.append((t_ms(), (bn, w, s)))
    except Exception as ex: print("  ERR", (bn, w, s), str(ex)[:80])
entry[(is16, dot)] = cur
for ms, c in sorted(res)[:6]: print("  %7.2f ms  %s" % (ms, c))
print("  ...worst %7.2f ms  %s" % max(res))
