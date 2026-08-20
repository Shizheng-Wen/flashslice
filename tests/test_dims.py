"""Dim-support contract of the fused slice/deslice kernels.

The kernels hold the whole head width D and slice count G in one tile, so
both must be powers of two in [16, 128]. Outside that range a layer built
with use_fused_slice=True must fall back to the eager path *visibly* --
never silently keep a flag that does nothing (cf. the inert-flag incident
documented in CLAUDE.md).
"""

import pytest
import torch

pytest.importorskip("triton")

from flashslice.kernels.fused_slice import unsupported_dims  # noqa: E402
from flashslice.layers.physics_attention import (  # noqa: E402
    Physics_Attention_Irregular_Mesh,
)


@pytest.mark.parametrize("d,g", [(16, 16), (32, 32), (32, 64), (32, 128),
                                 (128, 32)])
def test_supported_dims(d, g):
    assert unsupported_dims(d, g) is None


@pytest.mark.parametrize("d,g", [(32, 8), (32, 256), (32, 48), (8, 32),
                                 (256, 32)])
def test_unsupported_dims(d, g):
    why = unsupported_dims(d, g)
    assert why and "[16, 128]" in why


def _layer(g, d=32, heads=8):
    return Physics_Attention_Irregular_Mesh(
        dim=heads * d, heads=heads, dim_head=d, slice_num=g,
        use_fused_slice=True)


def test_layer_keeps_kernels_on_supported_dims():
    layer = _layer(g=32)
    assert layer.use_fused_slice is True
    assert layer.fused_slice_fallback is None


@pytest.mark.parametrize("g", [8, 256])
def test_layer_falls_back_visibly(g):
    """G=8 / G=256 (the ends of the paper's slice-count sweep) run eager."""
    layer = _layer(g=g)
    assert layer.use_fused_slice is False
    assert "slice_num=%d" % g in layer.fused_slice_fallback


def test_model_reports_effective_state():
    """The model, not just the layer, must report the EFFECTIVE state.

    A flag that is recorded but not applied silently trains a baseline replica.
    """
    from flashslice import Transolver

    def build(slice_num):
        return Transolver(space_dim=3, fun_dim=3, out_dim=1, n_hidden=64,
                          n_heads=2, n_layers=2, slice_num=slice_num,
                          mlp_ratio=1, use_fused_slice=True)

    assert build(32).use_fused_slice is True
    # unsupported G: the flag must read False on the built model
    assert build(8).use_fused_slice is False


def test_fused_rejects_incompatible_ablations():
    """The kernel never materializes the slice weights, so the two ablations
    that consume them must be refused at construction rather than silently
    producing a different model."""
    from flashslice import Transolver

    for flag in ("share_slice_across_layers", "slice_once"):
        with pytest.raises(ValueError, match="use_fused_slice"):
            Transolver(space_dim=3, fun_dim=3, out_dim=1, n_hidden=64, n_heads=2,
                       n_layers=2, slice_num=32, use_fused_slice=True, **{flag: True})


def test_forward_matches_eager_on_unsupported_dims():
    """The fallback must be an implementation switch only, not a model change."""
    torch.manual_seed(0)
    fused = _layer(g=8, d=32, heads=2)
    eager = Physics_Attention_Irregular_Mesh(
        dim=2 * 32, heads=2, dim_head=32, slice_num=8, use_fused_slice=False)
    eager.load_state_dict(fused.state_dict())
    x = torch.randn(1, 97, 64)
    with torch.no_grad():
        a = fused(x)[0]
        b = eager(x)[0]
    assert torch.equal(a, b)
