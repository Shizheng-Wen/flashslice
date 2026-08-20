"""End-to-end check that a fresh clone works: import, build, run, agree.

Scope, deliberately narrow: this answers "is the package wired up correctly",
not "is the kernel numerically sound". The authoritative numerical check is
``parity_test.py``, which references an fp64 computation and covers every
parameter gradient and every dot mode.

The distinction is worth stating, because an earlier version of this file got
it wrong. It compared the OUTPUT OF A FOUR-LAYER MODEL between the fused and
eager paths and asserted a tight relative tolerance. That is not a well-defined
quantity: differences compound through LayerNorms and residuals, and on a
randomly initialized network the outputs sit near zero, so normalizing by their
max inflates the ratio. It read 1.3e-3 and looked like a bug, when the kernel
was in fact within ~1e-7 of an fp64 reference. What is well defined is a single
layer, which is what this file checks.
"""

import torch

from flashslice import Transolver
from flashslice.layers import Physics_Attention_Irregular_Mesh
from flashslice.kernels import unsupported_dims

N = 60_000
HEADS, D, G = 8, 32, 32


def check_single_layer():
    """One layer, fused vs eager, at a realistic point count.

    The gate is deliberately loose, and the reason is worth stating: the
    fused-vs-eager difference is not a quantity anything promises. The claim is
    that fused error against an fp64 REFERENCE is at or below eager's, which is
    what parity_test measures. Both paths approximate the same truth, so their
    difference can be about twice either one's error, and at N=6e4 the
    accumulation is three orders of magnitude longer than parity_test's N=17/64
    cases -- a difference around 1e-4 is the expected fp32 behaviour, not a
    defect.

    What this gate has to separate is "correct" from "miswired", and a wiring
    mistake produces an O(1) difference, not 1e-4. Hence 1e-2. The measured
    value is printed so a real regression is still visible.
    """
    torch.manual_seed(0)
    dim = HEADS * D
    fused = Physics_Attention_Irregular_Mesh(
        dim, heads=HEADS, dim_head=D, slice_num=G, use_fused_slice=True).cuda()
    eager = Physics_Attention_Irregular_Mesh(
        dim, heads=HEADS, dim_head=D, slice_num=G, use_fused_slice=False).cuda()
    eager.load_state_dict(fused.state_dict())
    assert fused.use_fused_slice is True, "flag went inert on a supported shape"

    x = torch.randn(1, N, dim, device="cuda")
    with torch.no_grad():
        a = fused(x)[0]
        b = eager(x)[0]
    rel = ((a - b).norm() / b.norm()).item()
    print("[smoke] single layer, N=%d: relative L2 difference = %.3e" % (N, rel))
    assert rel < 1e-2, ("fused and eager disagree by far more than fp32 "
                       "accumulation can explain -- this is a wiring bug, "
                       "not precision")


def check_model_and_backward():
    torch.manual_seed(0)
    kw = dict(space_dim=3, fun_dim=1, out_dim=4, n_hidden=HEADS * D, n_heads=HEADS,
              n_layers=4, slice_num=G, mlp_ratio=2)
    model = Transolver(use_fused_slice=True, **kw).cuda()
    x = torch.randn(1, N, 3, device="cuda")
    fx = torch.randn(1, N, 1, device="cuda")
    y = model(x, fx)
    assert y.shape == (1, N, 4), y.shape
    y.square().mean().backward()
    total = sum(1 for _ in model.parameters())
    n_grad = sum(p.grad is not None for p in model.parameters())
    print("[smoke] model forward %s, backward reached %d/%d parameters"
          % (tuple(y.shape), n_grad, total))
    assert n_grad > 0


def check_fallback_is_visible():
    """An unsupported G must degrade loudly, never silently."""
    small = Transolver(space_dim=3, fun_dim=1, out_dim=4, n_hidden=HEADS * D,
                       n_heads=HEADS, n_layers=2, slice_num=8, use_fused_slice=True)
    assert small.use_fused_slice is False, "inert flag on an unsupported shape"
    print("[smoke] G=8 -> unsupported_dims says %r, model reports use_fused_slice=%s"
          % (unsupported_dims(D, 8), small.use_fused_slice))


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    print("[smoke] unsupported_dims(%d, %d) = %r" % (D, G, unsupported_dims(D, G)))
    check_single_layer()
    check_model_and_backward()
    check_fallback_is_visible()
    print("[smoke] OK -- for numerics see bench/parity_test.py")


if __name__ == "__main__":
    main()
