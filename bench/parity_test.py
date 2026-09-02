"""Parity gate for the fused slice/deslice kernels (kernels/slice_ops.py
and kernels/blocked.py).

fp32 forward/backward of the fused Physics_Attention_Irregular_Mesh against
an fp64 eager reference. The gate is purely relative: fused must sit as
close to the fp64 truth as eager does (<= 1.25x + 1e-6). No absolute cap —
the absolute error is set by the fp32 conditioning of the softmax-Jacobian
reductions, which ranges from ~1e-4 (outputs) through ~1.5e-3 (most grads
at N~1e5) up to ~4e-1 for the untied-deslice parameter grads, where
d_out @ z'^T is nearly constant across G and w*(dw - <dw,w>) is pure
cancellation; eager fp32 loses those digits identically. Also checks
bf16-autocast forward, run-to-run bitwise determinism (no atomics), the
no_token_attention / untied_deslice variants, ragged N, and G=64.

The G-blocked kernels get the same gate: forced onto the paper shape (where
the single-tile kernels are the default) so the two paths are held to the
same standard, and then on the shapes only they serve -- G=256 and 512, a
non-power-of-two G, G below 16, D=64 and 256, a non-power-of-two D, untied
deslice (which computes its own statistics), ragged N, and the bf16/tf32 dot
modes.

tf32x3 (every dot as three tf32 tensor-core products) is gated at 10x eager
like tf32: measured, it sits at 3-7x eager's fp32 error on outputs and
gradients (job 3264786) — fp32-class in spirit, two to three bits short in
fact, and 200x tighter than tf32. Its G=128 case is the one that exercises
the 8-warp tensor-core guard in the backward kernels.

Exits nonzero if any check fails. GPU required.
"""

import copy
import sys

import torch

from flashslice.layers.physics_attention import (
    Physics_Attention_Irregular_Mesh,
)

FAILED = []


def relerr(a, ref):
    ref = ref.double()
    n = ref.norm()
    d = (a.double() - ref).norm()
    return (d / n).item() if n > 0 else d.item()


def grads(model):
    return {k: (None if p.grad is None else p.grad.detach().clone())
            for k, p in model.named_parameters()}


def forward_backward(model, x, upstream):
    xx = x.clone().requires_grad_(True)
    out, _ = model(xx)
    out.backward(upstream.to(out.dtype))
    return out.detach(), xx.grad.detach(), grads(model)


def check(name, cond, detail=""):
    print("  [{}] {} {}".format("PASS" if cond else "FAIL", name, detail),
          flush=True)
    if not cond:
        FAILED.append(name)


def run_case(name, B, N, dim=256, heads=8, dim_head=32, G=32, gate=1.25,
             kernel_mode="auto", **kw):
    from flashslice.kernels import slice_ops as fs
    print("case: {} (B={}, N={}, D={}, G={}, kernels={}, {})".format(
        name, B, N, dim_head, G, kernel_mode, kw or "baseline"), flush=True)
    fs.set_kernel_mode(kernel_mode)
    try:
        _run_case(B, N, dim, heads, dim_head, G, gate, kw)
    finally:
        fs.set_kernel_mode("auto")


def _run_case(B, N, dim, heads, dim_head, G, gate, kw):
    torch.manual_seed(7)
    base = Physics_Attention_Irregular_Mesh(
        dim, heads=heads, dim_head=dim_head, dropout=0.0, slice_num=G,
        **kw).cuda()
    x = torch.randn(B, N, dim, device="cuda")
    up = torch.randn(B, N, dim, device="cuda")

    # fp64 eager reference
    m64 = copy.deepcopy(base).double()
    m64.use_fused_slice = False
    ref_out, ref_gx, ref_gp = forward_backward(m64, x.double(), up.double())

    runs = {}
    for fused in (False, True):
        m = copy.deepcopy(base)
        m.use_fused_slice = fused
        runs[fused] = forward_backward(m, x, up)
    out_e, gx_e, gp_e = runs[False]
    out_f, gx_f, gp_f = runs[True]

    pairs = [("out", out_e, out_f, ref_out), ("grad_x", gx_e, gx_f, ref_gx)]
    for pname, ref in ref_gp.items():
        if ref is None:
            check("gradNone:" + pname,
                  gp_e[pname] is None and gp_f[pname] is None)
            continue
        pairs.append(("grad:" + pname, gp_e[pname], gp_f[pname], ref))
    for label, te, tf, ref in pairs:
        ee, ef = relerr(te, ref), relerr(tf, ref)
        # Signal-dominated checks (strict fp32: ~1e-7..1e-5) must match within
        # `gate`. Where eager itself sits >1e-4 from the fp64 truth, the
        # quantity is cancellation noise in both implementations (the three
        # untied-deslice grads) and the noise-vs-noise ratio fluctuates: 3x.
        ok = ef <= gate * ee + 1e-6 or (ee > 1e-4 and ef <= 3.0 * ee)
        check(label, ok,
              "eager {:.3e}  fused {:.3e}  fused-vs-eager {:.3e}".format(
                  ee, ef, relerr(tf, te)))

    # bitwise determinism across two independent fused runs
    m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
    m1.use_fused_slice = m2.use_fused_slice = True
    r1 = forward_backward(m1, x, up)
    r2 = forward_backward(m2, x, up)
    det = torch.equal(r1[0], r2[0]) and torch.equal(r1[1], r2[1]) and all(
        torch.equal(r1[2][k], r2[2][k])
        for k in r1[2] if r1[2][k] is not None)
    check("determinism", det)

    # bf16-autocast forward sanity vs the fp64 reference
    me, mf = copy.deepcopy(base), copy.deepcopy(base)
    mf.use_fused_slice = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        oe, _ = me(x)
        of, _ = mf(x)
    ee, ef = relerr(oe.detach(), ref_out), relerr(of.detach(), ref_out)
    check("bf16_out", ef <= 1.5 * ee + 1e-3 and ef < 0.05,
          "eager {:.3e}  fused {:.3e}".format(ee, ef))


def run_bf16_case(name, dot, B=1, N=86840, dim=256, heads=8, dim_head=32,
                  G=32, kernel_mode="auto"):
    """bf16-autocast fwd+bwd: fused (given dot mode) vs eager, both against
    the fp64 reference. Everything here is bf16-class noise; the gate asks
    the fused error to stay within 2x of eager's (same class), plus bitwise
    determinism across two fused runs."""
    from flashslice.kernels import slice_ops as fs
    print("case: {} (bf16 autocast, dot={}, G={}, kernels={})".format(
        name, dot, G, kernel_mode), flush=True)
    fs.set_kernel_mode(kernel_mode)
    try:
        _run_bf16_case(dot, B, N, dim, heads, dim_head, G)
    finally:
        fs.set_kernel_mode("auto")


def _run_bf16_case(dot, B, N, dim, heads, dim_head, G):
    from flashslice.kernels import slice_ops as fs
    torch.manual_seed(7)
    base = Physics_Attention_Irregular_Mesh(
        dim, heads=heads, dim_head=dim_head, dropout=0.0, slice_num=G).cuda()
    x = torch.randn(B, N, dim, device="cuda")
    up = torch.randn(B, N, dim, device="cuda")
    m64 = copy.deepcopy(base).double()
    m64.use_fused_slice = False
    ref_out, ref_gx, ref_gp = forward_backward(m64, x.double(), up.double())

    def run_amp(m):
        xx = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = m(xx)
        out.backward(up.to(out.dtype))
        return out.detach(), xx.grad.detach(), grads(m)

    out_e, gx_e, gp_e = run_amp(copy.deepcopy(base))
    fs.set_dot_mode(dot)
    mf = copy.deepcopy(base)
    mf.use_fused_slice = True
    out_f, gx_f, gp_f = run_amp(mf)
    m2 = copy.deepcopy(base)
    m2.use_fused_slice = True
    r2 = run_amp(m2)
    fs.set_dot_mode("")

    pairs = [("out", out_e, out_f, ref_out), ("grad_x", gx_e, gx_f, ref_gx)]
    for pname, ref in ref_gp.items():
        if ref is not None:
            pairs.append(("grad:" + pname, gp_e[pname], gp_f[pname], ref))
    for label, te, tf, ref in pairs:
        ee, ef = relerr(te, ref), relerr(tf, ref)
        check(label, ef <= 2.0 * ee + 1e-6,
              "eager {:.3e}  fused {:.3e}".format(ee, ef))
    det = torch.equal(out_f, r2[0]) and torch.equal(gx_f, r2[1]) and all(
        torch.equal(gp_f[k], r2[2][k]) for k in gp_f if gp_f[k] is not None)
    check("determinism", det)


def main():
    print("torch {}  device {}".format(
        torch.__version__, torch.cuda.get_device_name(0)), flush=True)
    # Strict mode: both sides exact fp32 (fused dots follow this flag too).
    torch.backends.cuda.matmul.allow_tf32 = False
    run_case("baseline-team13N", B=1, N=86840)
    run_case("baseline-B2-ragged", B=2, N=10007)
    run_case("no-token-attn", B=1, N=50001, no_token_attention=True)
    run_case("untied", B=1, N=20011, untied_deslice=True)
    # every G the tile table is tuned for (powers of two in [16, 128]) — the
    # tiles differ per G, so each needs its own correctness check
    run_case("g16", B=1, N=30013, G=16)
    run_case("g64", B=1, N=30013, G=64)
    run_case("g128", B=1, N=30013, G=128)
    run_case("tiny-N17", B=2, N=17)
    run_case("one-block-N64", B=1, N=64)
    # The G-blocked kernels, first forced onto shapes the single-tile kernels
    # serve by default (same gate, same reference), then on their own shapes.
    blk = dict(kernel_mode="blocked")
    run_case("blk-paper-shape", B=1, N=86840, **blk)
    run_case("blk-B2-ragged", B=2, N=10007, **blk)
    run_case("blk-tiny-N17", B=2, N=17, G=256, **blk)
    run_case("blk-g256", B=1, N=30013, G=256, **blk)
    run_case("blk-g512", B=1, N=10007, G=512, **blk)
    run_case("blk-g48-masked", B=1, N=30013, G=48, **blk)
    run_case("blk-g8-padded", B=1, N=30013, G=8, **blk)
    run_case("blk-d64-g256", B=1, N=20011, dim=512, dim_head=64, G=256, **blk)
    run_case("blk-d256", B=1, N=10007, dim=1024, heads=4, dim_head=256, G=32,
             **blk)
    run_case("blk-d24-g40", B=1, N=20011, dim=192, dim_head=24, G=40, **blk)
    run_case("blk-untied-g256", B=1, N=20011, G=256, untied_deslice=True,
             **blk)
    # Opt-in tf32 value dots vs the ambient eager baseline. Eager's slice
    # einsums stay fp32 under allow_tf32 (skinny cuBLAS shapes), so fused-tf32
    # is measurably noisier than eager here by design (~3-4x, still ~5x below
    # the bf16 class); the gate only asserts the order of magnitude.
    torch.backends.cuda.matmul.allow_tf32 = True
    from flashslice.kernels import slice_ops as _fs
    _fs.set_dot_mode("tf32")
    run_case("tf32-optin", B=1, N=86840, gate=10.0)
    _fs.set_dot_mode("tf32")
    run_case("blk-tf32-optin-g256", B=1, N=30013, G=256, gate=10.0,
             kernel_mode="blocked")
    _fs.set_dot_mode("tf32x3")
    run_case("tf32x3-optin", B=1, N=86840, gate=10.0)
    run_case("tf32x3-g128", B=1, N=30013, G=128, gate=10.0)
    run_case("blk-tf32x3-g256", B=1, N=30013, G=256, gate=10.0,
             kernel_mode="blocked")
    _fs.set_dot_mode("")
    # bf16-native dots vs the eager bf16-autocast baseline (same class).
    run_bf16_case("bf16-value-dots", "bf16v")
    run_bf16_case("bf16-all-dots", "bf16")
    run_bf16_case("blk-bf16-value-dots-g256", "bf16v", N=30013, G=256,
                  kernel_mode="blocked")
    run_bf16_case("blk-bf16-all-dots-g256", "bf16", N=30013, G=256,
                  kernel_mode="blocked")
    run_bf16_case("blk-bf16-all-dots-paper", "bf16", kernel_mode="blocked")
    print("\n{} check(s) failed".format(len(FAILED)) if FAILED
          else "\nALL PARITY CHECKS PASSED")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
