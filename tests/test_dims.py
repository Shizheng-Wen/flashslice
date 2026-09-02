"""Dim-support contract of the fused slice/deslice kernels.

Two regimes. The single-tile kernels hold the whole head width D and slice
count G in one tile, so they serve both as powers of two in [16, 128]; the
G-blocked kernels serve any other G and D up to 256. Only D beyond that is
unsupported, and then a layer built with use_fused_slice=True must fall back
to the eager path *visibly* -- never silently keep a flag that does nothing
(see README, "Supported shapes").
"""

import pytest
import torch

pytest.importorskip("triton")

from flashslice.kernels.slice_ops import (  # noqa: E402
    single_tile_dims, unsupported_dims, set_kernel_mode, _use_blocked,
)
from flashslice.kernels.blocked import tiles  # noqa: E402
from flashslice.layers.physics_attention import (  # noqa: E402
    Physics_Attention_Irregular_Mesh,
)


@pytest.mark.parametrize("d,g", [(16, 16), (32, 32), (32, 64), (32, 128),
                                 (128, 32)])
def test_single_tile_dims(d, g):
    assert unsupported_dims(d, g) is None
    assert single_tile_dims(d, g)
    assert not _use_blocked(d, g)


@pytest.mark.parametrize("d,g", [(32, 8), (32, 256), (32, 48), (8, 32),
                                 (256, 32), (24, 40), (32, 1), (32, 1024)])
def test_blocked_dims(d, g):
    """Outside the single-tile range but inside the blocked one: served."""
    assert unsupported_dims(d, g) is None
    assert not single_tile_dims(d, g)
    assert _use_blocked(d, g)


@pytest.mark.parametrize("d,g", [(512, 32), (0, 32), (32, 0)])
def test_unsupported_dims(d, g):
    why = unsupported_dims(d, g)
    assert why and ("dim_head" in why or "slice_num" in why)


def test_kernel_mode_switch():
    try:
        set_kernel_mode("blocked")
        assert _use_blocked(32, 32)
        set_kernel_mode("single-tile")
        assert not _use_blocked(32, 32)
        with pytest.raises(ValueError, match="single-tile"):
            _use_blocked(32, 256)
        with pytest.raises(ValueError):
            set_kernel_mode("fast")
    finally:
        set_kernel_mode("auto")
    assert not _use_blocked(32, 32) and _use_blocked(32, 256)


@pytest.mark.parametrize("d,g,dt,gb", [
    (32, 256, 32, 64),    # G split into four blocks of 64
    (32, 48, 32, 64),     # one masked block
    (32, 8, 32, 16),      # G below the dot minimum: one padded block
    (24, 40, 32, 64),     # D padded to 32
    (64, 256, 64, 32),    # wider head, smaller G-block
    (256, 32, 256, 16),   # widest head: G-block at the minimum
])
def test_blocked_tiles(d, g, dt, gb):
    assert tiles(d, g) == (dt, gb)


def _layer(g, d=32, heads=8):
    return Physics_Attention_Irregular_Mesh(
        dim=heads * d, heads=heads, dim_head=d, slice_num=g,
        use_fused_slice=True)


@pytest.mark.parametrize("g", [32, 8, 256])
def test_layer_keeps_kernels_on_supported_dims(g):
    layer = _layer(g=g)
    assert layer.use_fused_slice is True
    assert layer.fused_slice_fallback is None


def test_layer_falls_back_visibly():
    """D=512 exceeds what any kernel holds in one tile: eager, and it says so."""
    layer = _layer(g=32, d=512, heads=2)
    assert layer.use_fused_slice is False
    assert "dim_head=512" in layer.fused_slice_fallback


def test_model_reports_effective_state():
    """The model, not just the layer, must report the EFFECTIVE state.

    A flag that is recorded but not applied silently trains a baseline replica.
    """
    from flashslice import Transolver

    def build(dim_head):
        return Transolver(space_dim=3, fun_dim=3, out_dim=1, n_hidden=64,
                          n_heads=2, n_layers=2, slice_num=32, mlp_ratio=1,
                          dim_head=dim_head, use_fused_slice=True)

    assert build(32).use_fused_slice is True
    # unsupported D: the flag must read False on the built model
    assert build(512).use_fused_slice is False


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
    fused = _layer(g=8, d=512, heads=2)
    eager = Physics_Attention_Irregular_Mesh(
        dim=2 * 512, heads=2, dim_head=512, slice_num=8, use_fused_slice=False)
    eager.load_state_dict(fused.state_dict())
    x = torch.randn(1, 97, 1024)
    with torch.no_grad():
        a = fused(x)[0]
        b = eager(x)[0]
    assert torch.equal(a, b)
