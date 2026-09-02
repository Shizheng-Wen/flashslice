"""Turn `bench_kernels.py --family blocked` sweep results into a _CFG_BLK entry.

    python bench/pick_tiles.py bench/results/sweep_blk_g256_fp32.json \\
                              bench/results/sweep_blk_g256_bf16.json

Reads the results of each JSON (one sweep = one G, one dtype, any dot
levels), groups by (G_block, D_tile) and prints the dict literal to paste
into flashslice/kernels/blocked.py. When several N values were swept the
winner is the config with the lowest mean slowdown against the per-N best,
the rule the single-tile tables were built with. Pure Python: no torch, no
GPU, safe on a login node.
"""

import json
import re
import sys
from collections import defaultdict

KERNELS = ("stats", "slice_fwd_g", "deslice_fwd_n", "slice_bwd_n",
           "slice_bwd_g", "deslice_bwd_n", "deslice_bwd_g")
DOT_LEVEL = {"ieee": 0, "tf32": 1, "bf16v": 2, "bf16": 3}


def _tiles(d, g):
    """blocked.tiles for sweeps written before the JSON recorded them."""
    p2 = lambda v: max(16, 1 << (v - 1).bit_length())  # noqa: E731
    dt = p2(d)
    return dt, min(max(16, min(64, 2048 // dt)), p2(g))


def main(paths):
    tables = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        dims = data["dims"]
        if dims.get("family") != "blocked":
            raise SystemExit("%s is not a --family blocked sweep" % path)
        if "G_block" in dims:
            dt, gb = dims["D_tile"], dims["G_block"]
        else:
            dt, gb = _tiles(dims["D"], dims["G"])
        per_cfg = defaultdict(lambda: defaultdict(dict))  # (k, dtype, dot) -> cfg -> {N: ms}
        ns = set()
        for key0, entries in data["results"].items():
            n, dt_s = key0.split("::")
            ns.add(n)
            for tag, r in entries.items():
                if not isinstance(r, dict) or "ms" not in r or tag.count("/") != 2:
                    continue  # ERR entry, or a --defaults-only table
                kname, cfg, dot = tag.split("/")
                per_cfg[(kname, dt_s, dot)][cfg][n] = r["ms"]
        for (kname, dt_s, dot), cfgs in per_cfg.items():
            complete = {c: v for c, v in cfgs.items() if len(v) == len(ns)}
            if not complete:
                continue
            best_per_n = {n: min(v[n] for v in complete.values()) for n in ns}
            score = {c: sum(v[n] / best_per_n[n] for n in ns) / len(ns)
                     for c, v in complete.items()}
            cfg = min(score, key=score.get)
            bn, w, st = (int(x) for x in re.match(r"bn(\d+)w(\d+)s(\d+)", cfg).groups())
            tables[(gb, dt)][kname][(dt_s == "bf16", DOT_LEVEL[dot])] = (bn, w, st)
    for (gb, dt), kernels in sorted(tables.items()):
        print("    (%d, %d): {" % (gb, dt))
        for kname in KERNELS:
            if kname not in kernels:
                continue
            print("        %r: {" % kname)
            for key in sorted(kernels[kname]):
                print("            %r: %r," % (key, kernels[kname][key]))
            print("        },")
        print("    },")


if __name__ == "__main__":
    main(sys.argv[1:])
