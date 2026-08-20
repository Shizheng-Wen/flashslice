"""Physics-attention for irregular meshes, with the paper's ablation flags.

Only the irregular-mesh module is shipped. The reference implementation also
has structured 1D/2D/3D variants; every experiment in the paper runs on
unstructured point clouds, and the fused kernel supports only this path, so
shipping the others would be untested code.

Set ``use_fused_slice=True`` to route slice/deslice through the Triton
kernels in ``flashslice.kernels``. It is an implementation switch, not an
ablation: the outputs are the same and checkpoints interchange.
"""

import torch.nn as nn
import torch
from einops import rearrange
import logging

logger = logging.getLogger(__name__)


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    # Ablation flags (defaults reproduce the original module exactly):
    #   untied_deslice: deslice uses its own projection/temperature instead of reusing slice_weights
    #   no_token_attention: skip attention among slice tokens (keeps to_v transform only)
    # Width decoupling (advisor's memory study):
    #   dim_head: full-resolution per-head width -> drives the heavy [B,N,heads*dim_head]
    #     activations (in_project_fx/x, slice_token, deslice, to_out). Smaller = less memory.
    #   slice_dim_head: width of the slice-token attention (on G tokens only, ~free in memory).
    #     None -> = dim_head (exact original). Lets a narrow full-res path keep a wide latent.
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None,
                 untied_deslice=False, no_token_attention=False, slice_dim_head=None,
                 use_fused_slice=False):
        super().__init__()
        self.use_fused_slice = use_fused_slice
        self.fused_slice_fallback = None
        if use_fused_slice:
            # Decide at construction, not mid-training. The kernels tile the
            # whole D and G axes, so both must be powers of two in [16, 128]
            # (see fused_slice.unsupported_dims). Outside that range — e.g. the
            # G=8 / G=256 ends of the slice-count sweep — this layer runs the
            # eager path instead of failing the run. The fallback is loud and
            # is recorded in self.fused_slice_fallback / self.use_fused_slice,
            # so an inert flag stays visible on the built model.
            from ..kernels.fused_slice import unsupported_dims
            why = unsupported_dims(dim_head, slice_num)
            if why:
                warn = logger.warning
                warn("use_fused_slice=True but %s; falling back to the eager "
                     "slice/deslice path (identical outputs, baseline speed "
                     "and memory)." % why)
                self.use_fused_slice = False
                self.fused_slice_fallback = why
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.slice_dim_head = slice_dim_head if slice_dim_head is not None else dim_head
        self.heads = heads
        self.scale = self.slice_dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.untied_deslice = untied_deslice
        self.no_token_attention = no_token_attention

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        if untied_deslice:
            self.deslice_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
            self.in_project_deslice = nn.Linear(dim_head, slice_num)
            torch.nn.init.orthogonal_(self.in_project_deslice.weight)
        # slice-token attention runs at slice_dim_head; project back to dim_head for deslice.
        self.to_q = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, self.slice_dim_head, bias=False)
        self.slice_down = (
            nn.Identity() if self.slice_dim_head == dim_head
            else nn.Linear(self.slice_dim_head, dim_head, bias=False)
        )
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def _forward_fused(self, x):
        """Fused slice/deslice (see layers/fused_slice.py). Same math as the
        eager path; w (B,H,N,G) is never materialized and the backward
        recomputes it, so the permute/contiguous copies and the stored
        slice-weight activations disappear."""
        from ..kernels.fused_slice import fused_slice, fused_deslice
        B, N, C = x.shape
        H, D = self.heads, self.dim_head

        fx_mid = self.in_project_fx(x).view(B, N, H, D)
        x_mid = self.in_project_x(x).view(B, N, H, D)
        tau = self.temperature.view(H)
        z_num, s = fused_slice(x_mid, fx_mid, self.in_project_slice.weight,
                               self.in_project_slice.bias, tau)
        slice_token = z_num / (s + 1e-5).unsqueeze(-1)  # B H G dim_head

        ### (2) Attention among slice tokens — unchanged, tiny (G tokens)
        if self.no_token_attention:
            out_slice_token = self.to_v(slice_token)
        else:
            q_slice_token = self.to_q(slice_token)
            k_slice_token = self.to_k(slice_token)
            v_slice_token = self.to_v(slice_token)
            dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
            attn = self.dropout(self.softmax(dots))
            out_slice_token = torch.matmul(attn, v_slice_token)
        out_slice_token = self.slice_down(out_slice_token)

        ### (3) Deslice
        if self.untied_deslice:
            out_x = fused_deslice(x_mid, self.in_project_deslice.weight,
                                  self.in_project_deslice.bias,
                                  self.deslice_temperature.view(H), out_slice_token)
        else:
            out_x = fused_deslice(x_mid, self.in_project_slice.weight,
                                  self.in_project_slice.bias, tau, out_slice_token)
        # out_x is (B, N, H, D) already — reshape is a view, no copy.
        return self.to_out(out_x.reshape(B, N, H * D)), None

    def forward(self, x, slice_weights_in=None):
        # B N C; slice_weights_in: externally provided slice weights (share-across-layers ablation)
        if self.use_fused_slice:
            if slice_weights_in is not None:
                raise ValueError("use_fused_slice cannot consume externally "
                                 "provided slice weights.")
            return self._forward_fused(x)
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        if slice_weights_in is None or self.untied_deslice:
            x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
                .permute(0, 2, 1, 3).contiguous()  # B H N C
        if slice_weights_in is None:
            slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        else:
            slice_weights = slice_weights_in
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens (at slice_dim_head width, on G tokens)
        if self.no_token_attention:
            out_slice_token = self.to_v(slice_token)  # B H G slice_dim_head, no inter-token mixing
        else:
            q_slice_token = self.to_q(slice_token)
            k_slice_token = self.to_k(slice_token)
            v_slice_token = self.to_v(slice_token)
            dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
            attn = self.softmax(dots)
            attn = self.dropout(attn)
            out_slice_token = torch.matmul(attn, v_slice_token)  # B H G slice_dim_head
        out_slice_token = self.slice_down(out_slice_token)  # B H G dim_head (Identity if equal)

        ### (3) Deslice
        if self.untied_deslice:
            deslice_weights = self.softmax(self.in_project_deslice(x_mid) / self.deslice_temperature)
        else:
            deslice_weights = slice_weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, deslice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x), slice_weights
