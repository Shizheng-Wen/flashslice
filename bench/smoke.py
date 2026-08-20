"""End-to-end check that the extracted package works: import, build, run, and
agree with the eager path.

This is what a fresh clone should be able to do. Run it on a GPU node.
"""

import torch

from flashslice import Transolver
from flashslice.kernels import unsupported_dims


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    torch.manual_seed(0)
    kw = dict(space_dim=3, fun_dim=1, out_dim=4, n_hidden=256, n_heads=8,
              n_layers=4, slice_num=32, mlp_ratio=2)

    fused = Transolver(use_fused_slice=True, **kw).cuda()
    eager = Transolver(use_fused_slice=False, **kw).cuda()
    eager.load_state_dict(fused.state_dict())
    assert fused.use_fused_slice is True, "flag went inert on a supported shape"
    print("[smoke] built both models; unsupported_dims(32, 32) =",
          unsupported_dims(32, 32))

    x = torch.randn(1, 60_000, 3, device="cuda")
    fx = torch.randn(1, 60_000, 1, device="cuda")

    with torch.no_grad():
        a, b = fused(x, fx), eager(x, fx)
    rel = ((a - b).abs().max() / b.abs().max()).item()
    print("[smoke] forward: max rel diff fused vs eager = %.3e" % rel)
    assert rel < 1e-5, "fused and eager disagree"

    # backward through the kernel, and a parameter gradient comparison
    loss_f = fused(x, fx).square().mean(); loss_f.backward()
    loss_e = eager(x, fx).square().mean(); loss_e.backward()
    worst, name = 0.0, ""
    for (n, pf), (_, pe) in zip(fused.named_parameters(), eager.named_parameters()):
        if pf.grad is None:
            continue
        d = (pf.grad - pe.grad).abs().max() / pe.grad.abs().max().clamp_min(1e-12)
        if d.item() > worst:
            worst, name = d.item(), n
    print("[smoke] backward: worst param-grad rel diff = %.3e (%s)" % (worst, name))
    assert worst < 1e-4, "gradients disagree"

    # the fallback must be visible, not silent
    small = Transolver(use_fused_slice=True, **{**kw, "slice_num": 8})
    assert small.use_fused_slice is False, "inert flag on an unsupported shape"
    print("[smoke] G=8 falls back visibly: use_fused_slice ->", small.use_fused_slice)
    print("[smoke] OK")


if __name__ == "__main__":
    main()
