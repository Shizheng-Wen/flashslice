# API reference

## Ops

!!! note
    Signatures as of `main`. Tag `v0.1.0` lacks the `stats` argument of
    `fused_slice`, the `return_stats` argument of `fused_deslice`, weight
    layouts other than `(G, D)`, and a value width apart from `D`.

### `fused_slice(x_mid, fx_mid, weight, bias, tau, return_stats=False, stats=None)`

Pool point features onto slots.

| argument | shape | notes |
| --- | --- | --- |
| `x_mid` | `(B, N, H, D)` | fp32, or bf16 for the 16-bit dot modes |
| `fx_mid` | `(B, N, H, DV)` | same dtype as `x_mid` |
| `weight` | `(G, D)`, `(H, G, D)`, `(B, H, G, D)` | slot projection; gradient returned in this shape |
| `bias` | `None`, `(G,)`, `(H, G)`, `(B, H, G)` | added before the temperature |
| `tau` | `(H,)` | per-head temperature, learned; fp32 |
| `stats` | `(B, H, 2, N)` fp32 or `None` | statistics from a tied deslice on the same `x_mid`, `weight`, `bias`, `tau`; skips the statistics pass |

Returns fp32 `z_num (B, H, G, DV)` and `s (B, H, G)`; normalize outside as
`z = z_num / (s + eps)[..., None]`. With `return_stats=True` a third value:
the statistics on the blocked path (detached), `None` on the single-tile
path. Either is accepted by `fused_deslice`.

### `fused_deslice(x_mid, weight, bias, tau, tokens, stats=None, return_stats=False)`

Broadcast slot values back to points. `tokens` is `(B, H, G, DV)`; returns
`out (B, N, H, DV)` in `x_mid`'s dtype, already in the layout the output
projection expects after a reshape. `stats` from a tied slice skips the
statistics; with none given the blocked path forms `out` and the statistics
in one online pass, and `return_stats=True` returns `(out, stats)` for a
tied slice that follows (`(out, None)` on the single-tile path).

### Shape and mode functions

| function | what it does |
| --- | --- |
| `unsupported_dims(D, G, DV=None)` | reason string when no family serves the shape, else `None` |
| `single_tile_dims(D, G, DV=None)` | `True` when the single-tile kernels serve it by default |
| `set_dot_mode(mode)` | `"ieee"` `"tf32"` `"bf16v"` `"bf16"` `"tf32x3"`; also `FLASHSLICE_DOT_MODE` |
| `set_kernel_mode(mode)` | `"auto"` `"single-tile"` `"blocked"`; also `FLASHSLICE_KERNEL_MODE` |
| `set_stats_mode(mode)`, `stats_mode()` | `"online"` (default) or `"two-pass"`; also `FLASHSLICE_STATS_MODE` |
| `set_block_g(gb)` | force the blocked family's `G` block size (power of two ≥ 16), `None` for automatic |
| `blocked.compute_stats(x_mid, weight, bias, tau, dot=0)` | the statistics alone, `(B, H, 2, N)` fp32 |
| `blocked.tiles(D, G)` | `(D_tile, G_block)` the blocked family would use |

All ops are `torch.library` custom ops (`flashslice::flash_slice`,
`flash_deslice`, `slice_blk`, `deslice_blk` and their `_bwd` twins) with
autograd and fake registrations, so they compose with `torch.compile` as
opaque nodes.

## The layer and the model

`flashslice.layers.Physics_Attention_Irregular_Mesh(dim, heads, dim_head,
dropout, slice_num, use_fused_slice=False, no_token_attention=False,
untied_deslice=False, ...)` is Transolver's layer with the slice/deslice
routed through the kernels when `use_fused_slice=True`, and
`flashslice.Transolver(space_dim, fun_dim, out_dim, n_hidden, n_heads,
n_layers, slice_num, ..., use_fused_slice=False, **ablation)` the model.
Every variant in the paper is one flag, at most one at a time:

| flag | what changes |
| --- | --- |
| `no_token_attention` | attention among slice tokens becomes a per-token linear map — tokens stop interacting |
| `untie_slice_weights` | deslice gets its own projection and temperature |
| `share_slice_across_layers` | slice weights computed once at layer 1, reused by all layers |
| `slice_once` | slice once → deep transformer on `G` tokens → deslice once (Perceiver limit) |
| `mlp_only` | the attention sublayer is removed entirely (pointwise lower bound) |

`use_fused_slice` is orthogonal to all of them, except that it is refused
together with `share_slice_across_layers` and `slice_once`: both consume the
slice weights the kernel never materializes, so they raise at construction
rather than quietly producing a different model. A built layer or model
reports its *effective* state: `use_fused_slice` reads `False` after a
fallback and `fused_slice_fallback` says why.
