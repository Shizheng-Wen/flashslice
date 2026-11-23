# Copyright 2026 Shizheng Wen
# SPDX-License-Identifier: Apache-2.0

"""The Hopper tile tables per Triton series: selection and shape.

CPU-only: nothing here compiles a kernel. What a table costs is measured by
bench/bench_kernels.py --defaults-only, not asserted.
"""

import pytest
import torch

from flashslice.kernels import blocked as fb
from flashslice.kernels import set_tile_series, set_tile_table, tile_series
from flashslice.kernels import slice_ops as fs

KERNELS = ("slice_fwd", "deslice_fwd", "slice_bwd", "deslice_bwd")
GS = (16, 32, 64, 128)


@pytest.fixture
def hopper():
    """Pin the Hopper class whatever the GPU; restore both switches after."""
    set_tile_table("hopper")
    try:
        yield
    finally:
        set_tile_table("auto")
        set_tile_series("auto")


@pytest.mark.parametrize("version,series", [
    ("3.0.0", "3.0"), ("3.0", "3.0"), ("3.1.0", "3.0"), ("3.4.0", "3.0"), ("3.5.1", "3.0"),
    ("3.6.0", "3.6"), ("3.6.0+git9844da95", "3.6"), ("3.7.0", "3.6"), ("4.0.0", "3.6"),
])
def test_series_of_a_version(version, series):
    assert tile_series(version) == series


def test_series_override():
    try:
        set_tile_series("3.0")
        assert tile_series() == "3.0"
        set_tile_series("3.6")
        assert tile_series() == "3.6"
        with pytest.raises(ValueError):
            set_tile_series("3.3")
    finally:
        set_tile_series("auto")
    import triton
    assert tile_series() == tile_series(triton.__version__)


def _is_tile(t):
    return (len(t) == 3 and all(isinstance(v, int) for v in t)
            and t[0] in (16, 32, 64, 128, 256) and t[1] in (4, 8) and t[2] in (1, 2, 3))


@pytest.mark.parametrize("table", [fs._CFG, fs._CFG_T36, fs._CFG_TC_T36])
def test_single_tile_tables_are_complete(table):
    assert tuple(sorted(table)) == GS
    for g in GS:
        assert set(table[g]) == {(k, is16) for k in KERNELS for is16 in (False, True)}
        assert all(_is_tile(t) for t in table[g].values())


@pytest.mark.parametrize("table", [fs._CFG_BF16, fs._CFG_BF16_T36])
def test_bf16_tables_are_complete(table):
    assert tuple(sorted(table)) == GS
    for g in GS:
        assert set(table[g]) == set(KERNELS)
        assert all(_is_tile(t) for t in table[g].values())


def test_triton_36_tensor_core_entries_run_as_written():
    """The 3.6 tensor-core entries were picked among what the launch code lets
    through, so _cfg returns them unchanged: one stage for tf32, 4 warps on
    the backward kernels and at G = 16."""
    for g in GS:
        for (k, _), (bn, w, s) in fs._CFG_TC_T36[g].items():
            assert s == 1
            assert w == 4 or not (g <= 16 or k.endswith("_bwd"))
        for k, (bn, w, s) in fs._CFG_BF16_T36[g].items():
            assert w == 4 or not (g <= 16 or k.endswith("_bwd"))


def test_blocked_triton_36_table_is_well_formed():
    table = fb._CFG_BLK_T36
    assert {(32, 32), (64, 32), (64, 64, 32)} <= set(table)
    for kernels in table.values():
        assert set(kernels) <= set(fb._FAMILY)
        for entries in kernels.values():
            for (is16, dot), t in entries.items():
                assert isinstance(is16, bool) and dot in (0, 1, 3)
                assert _is_tile(t)


def test_cfg_reads_the_table_of_the_series(hopper):
    """On Hopper: under 3.6, ieee reads _CFG_T36, the tensor-core value dots
    _CFG_TC_T36 and bf16 dots _CFG_BF16_T36; under 3.0 the tables as they
    were."""
    fp32 = torch.empty(0, dtype=torch.float32)
    set_tile_series("3.6")
    assert fs._cfg("slice_fwd", fp32, 0, 32) == fs._CFG_T36[32][("slice_fwd", False)]
    assert fs._cfg("slice_fwd", fp32, 1, 32) == fs._CFG_TC_T36[32][("slice_fwd", False)]
    assert fs._cfg("slice_fwd", fp32, 4, 32) == fs._CFG_TC_T36[32][("slice_fwd", False)]
    assert fs._cfg("slice_fwd", fp32, 3, 32) == fs._CFG_BF16_T36[32]["slice_fwd"]
    assert fb._cfg_blk("deslice_bwd_n", True, 0, 64, 32) == fb._CFG_BLK_T36[(64, 32)]["deslice_bwd_n"][(True, 0)]
    set_tile_series("3.0")
    assert fs._cfg("slice_fwd", fp32, 0, 32) == (256, 4, 1)
    assert fs._cfg("slice_fwd", fp32, 1, 32) == (256, 4, 1)
    assert fs._cfg("deslice_bwd", fp32, 0, 128) == (64, 8, 2)
    assert fs._cfg("deslice_bwd", fp32, 1, 128) == (64, 4, 2)   # 8 warps clamped under a tensor-core dot
    assert fs._cfg("slice_fwd", fp32, 3, 32) == (256, 4, 3)
    assert fb._cfg_blk("deslice_bwd_n", True, 0, 64, 32) == fb._CFG_BLK[(64, 32)]["deslice_bwd_n"][(True, 0)]


def test_ada_tables_do_not_follow_the_series():
    """The Ada class has one set, whatever the series."""
    fp32 = torch.empty(0, dtype=torch.float32)
    set_tile_table("ada")
    try:
        set_tile_series("3.0")
        a = fs._cfg("slice_fwd", fp32, 0, 32)
        set_tile_series("3.6")
        assert fs._cfg("slice_fwd", fp32, 0, 32) == a == fs._CFG_ADA[32][("slice_fwd", False)]
    finally:
        set_tile_table("auto")
        set_tile_series("auto")
