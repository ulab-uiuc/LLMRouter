"""Hidden trajectory encoders and the MHLC capability head."""

import math
from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn


def no_amp_fp32(enabled: bool = True):
    """Return a context manager that disables AMP to run a block in fp32 for numerical stability."""
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=False)


def _num_groups(c: int, g: int = 8) -> int:
    """Choose a valid number of groups for GroupNorm / grouped convs."""
    for k in [g, 6, 4, 3, 2, 1]:
        if c % k == 0:
            return k
    return 1


def _module_param_dtype(mod: nn.Module) -> torch.dtype:
    for p in mod.parameters():
        return p.dtype
    return torch.float32


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        B, Tq, D = Q.shape
        Tk = K.size(1)

        q = self.q(Q).view(B, Tq, self.h, self.dk).transpose(1, 2)  # (B,h,Tq,dk)
        k = self.k(K).view(B, Tk, self.h, self.dk).transpose(1, 2)  # (B,h,Tk,dk)
        v = self.v(V).view(B, Tk, self.h, self.dk).transpose(1, 2)  # (B,h,Tk,dk)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)  # (B,Tq,D)
        return self.o(out)


class MAB(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, ff_mult: int = 4
    ):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, pdrop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        x = self.ln1(Q + self.mha(Q, K, K))
        x = self.ln2(x + self.ff(x))
        return x


class SAB(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, num_layers: int = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [MAB(d_model, n_heads, pdrop) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for mab in self.layers:
            X = mab(X, X)
        return X


class PMA(nn.Module):
    """Pooling by Multi-head Attention (Set Transformer)."""

    def __init__(
        self, d_model: int, num_seeds: int = 4, n_heads: int = 4, pdrop: float = 0.1
    ):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, d_model) / math.sqrt(d_model))
        self.mab = MAB(d_model, n_heads, pdrop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)  # (B,K,d)
        return self.mab(S, X)  # (B,K,d)


class HiddenFeatureExtractorLite(nn.Module):
    """
    Features from hidden-state sequences via gated dilated 1D convs + SAB + PMA.
    """

    def __init__(
        self,
        D_model: int,
        D_HID: int = 256,
        d_tok: int = 192,
        k_hid: int = 192,
        groups: int = 8,
        K: int = 3,
        sab_layers: int = 3,
        sab_heads: int = 4,
        pdrop: float = 0.10,
    ):
        super().__init__()
        self.k_hid, self.d_tok = int(k_hid), int(d_tok)
        self.norm, self.proj = nn.LayerNorm(D_model), nn.Linear(D_model, d_tok)

        g = _num_groups(d_tok)  # ensure divisibility

        def dw_block(dil):
            return nn.Sequential(
                nn.Conv1d(d_tok, d_tok, 5, padding=2 * dil, dilation=dil, groups=g),
                nn.GroupNorm(_num_groups(d_tok), d_tok),
                nn.GELU(),
            )

        self.dw1, self.dw2, self.dw3 = dw_block(1), dw_block(2), dw_block(4)
        self.gate = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)
        self.se1d = nn.Sequential(
            nn.Conv1d(d_tok, max(8, d_tok // 8), 1),
            nn.GELU(),
            nn.Conv1d(max(8, d_tok // 8), d_tok, 1),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(pdrop)
        self.pos = nn.Parameter(torch.randn(1, self.k_hid, d_tok) / math.sqrt(d_tok))
        self.sab = SAB(
            d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers
        )
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok, 2 * d_tok),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_HID),
        )

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        _B, _S, _ = last_hidden.shape
        x = self.proj(self.norm(last_hidden)).permute(0, 2, 1).contiguous()

        # Mask-free downsampling (fp32 for pooling only)
        with no_amp_fp32(True):
            x_ds = F.adaptive_avg_pool1d(x.to(torch.float32), self.k_hid)
        x_ds = x_ds.to(_module_param_dtype(self.dw1))  # match conv dtype

        y1, y2, y3 = self.dw1(x_ds), self.dw2(x_ds), self.dw3(x_ds)
        g = torch.softmax(self.gate, dim=0)
        mix = g[0] * y1 + g[1] * y2 + g[2] * y3
        z = self.drop(mix * self.se1d(mix) + x_ds)

        tok = z.permute(0, 2, 1).contiguous() + self.pos.to(z.dtype)
        tok = self.sab(tok)
        pooled = self.pma(tok).flatten(1)

        return self.out(pooled)


class CorrectnessHeadLite(nn.Module):
    """
    Fuse (attention, confidence, hidden-state) feature vectors with learned gating.
    Outputs *logits* (no sigmoid/softmax here).
    """

    def __init__(
        self,
        D_ATT: int,
        D_CONF: int,
        D_HID: int,
        use_attn=True,
        use_conf=True,
        use_hid=True,
        pdrop: float = 0.10,
        num_labels: int = 4,  # <-- NEW (default 4)
    ):
        super().__init__()
        self.use_attn, self.use_conf, self.use_hid = use_attn, use_conf, use_hid
        self.num_labels = num_labels  # <-- NEW

        dims = []
        if use_attn:
            dims.append(D_ATT)
        if use_conf:
            dims.append(D_CONF)
        if use_hid:
            dims.append(D_HID)
        D = sum(dims)
        if D == 0:
            raise ValueError("Enable at least one modality.")

        self.g_att = (
            nn.Sequential(nn.LayerNorm(D_ATT), nn.Linear(D_ATT, 1))
            if use_attn
            else None
        )
        self.g_con = (
            nn.Sequential(nn.LayerNorm(D_CONF), nn.Linear(D_CONF, 1))
            if use_conf
            else None
        )
        self.g_hid = (
            nn.Sequential(nn.LayerNorm(D_HID), nn.Linear(D_HID, 1)) if use_hid else None
        )

        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 384),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(128, num_labels),  # <-- CHANGED (was 128 -> 1)
        )

    def forward(
        self,
        z_att: torch.Tensor | None,
        z_conf: torch.Tensor | None,
        z_hid: torch.Tensor | None,
        return_penultimate: bool = False,
    ):
        # Optional sanity checks (recommended)
        if self.use_attn:
            assert z_att is not None, "use_attn=True but z_att is None"
        if self.use_conf:
            assert z_conf is not None, "use_conf=True but z_conf is None"
        if self.use_hid:
            assert z_hid is not None, "use_hid=True but z_hid is None"

        chunks, gates = [], []
        if self.use_attn:
            chunks.append(z_att)
            gates.append(self.g_att(z_att))  # (B,1)
        if self.use_conf:
            chunks.append(z_conf)
            gates.append(self.g_con(z_conf))  # (B,1)
        if self.use_hid:
            chunks.append(z_hid)
            gates.append(self.g_hid(z_hid))  # (B,1)

        g = torch.softmax(torch.cat(gates, dim=-1), dim=-1)  # (B, M)
        out_slices = [ch * g[:, i : i + 1] for i, ch in enumerate(chunks)]
        x = torch.cat(out_slices, dim=-1)

        x_norm = self.ln(x)

        h_last = self.mlp[:-1](x_norm)  # (B, 128)
        logits = self.mlp[-1](h_last)  # (B, num_labels)  <-- CHANGED

        if return_penultimate:
            return logits, h_last
        return logits


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def get_slopes_power_of_2(n: int) -> torch.Tensor:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return torch.tensor([start * (ratio**i) for i in range(n)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)

    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = _get_alibi_slopes(2 * closest_power_of_2)[0::2][
        : n_heads - closest_power_of_2
    ]
    return torch.cat([slopes, extra], dim=0)


def _as_list_of_multilayer_sequences(
    hidden: torch.Tensor | list[torch.Tensor],
    lengths: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Preferred:
        hidden = [Tensor[M, S_i, D], Tensor[M, S_j, D], ...]

    Also supports:
        hidden = Tensor[B, M, S, D] with optional lengths[B]
    """
    if isinstance(hidden, list):
        out = []
        for i, x in enumerate(hidden):
            if not torch.is_tensor(x) or x.dim() != 3:
                raise ValueError(
                    f"hidden[{i}] must have shape [M, S, D], got {type(x)} / {getattr(x, 'shape', None)}"
                )
            out.append(x)
        return out

    if not torch.is_tensor(hidden) or hidden.dim() != 4:
        raise ValueError("hidden must be a List[Tensor[M,S,D]] or Tensor[B,M,S,D].")

    B, _M, S, _D = hidden.shape
    if lengths is None:
        return [hidden[b] for b in range(B)]

    if lengths.dim() != 1 or lengths.numel() != B:
        raise ValueError("lengths must have shape [B].")

    seqs = []
    for b in range(B):
        L = int(lengths[b].item())
        L = max(1, min(L, S))
        seqs.append(hidden[b, :, :L, :])
    return seqs


def _as_list_of_layer_ids(
    layer_ids: torch.Tensor | Sequence[int] | list[torch.Tensor | Sequence[int]] | None,
    seqs: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Returns one LongTensor[M] per example.

    Supports:
      - None
      - Tensor[M]
      - Tensor[B, M]
      - list[int] shared across batch
      - list[tensor/list[int]] per example
    """
    if layer_ids is None:
        return [
            torch.arange(seq.size(0), device=seq.device, dtype=torch.long)
            for seq in seqs
        ]

    if torch.is_tensor(layer_ids):
        if layer_ids.dim() == 1:
            common = layer_ids.long()
            out = []
            for seq in seqs:
                if common.numel() != seq.size(0):
                    raise ValueError(
                        f"layer_ids has {common.numel()} ids but sequence has M={seq.size(0)} layers."
                    )
                out.append(common.to(seq.device))
            return out

        if layer_ids.dim() == 2:
            if layer_ids.size(0) != len(seqs):
                raise ValueError(
                    "layer_ids[B,M] first dimension must match batch size."
                )
            out = []
            for i, seq in enumerate(seqs):
                ids_i = layer_ids[i].long()
                if ids_i.numel() != seq.size(0):
                    raise ValueError(
                        f"layer_ids[{i}] has {ids_i.numel()} ids but sequence has M={seq.size(0)} layers."
                    )
                out.append(ids_i.to(seq.device))
            return out

        raise ValueError("Tensor layer_ids must be shape [M] or [B, M].")

    if isinstance(layer_ids, (list, tuple)):
        if len(layer_ids) == 0:
            raise ValueError("layer_ids list is empty.")

        if isinstance(layer_ids[0], int):
            common = torch.tensor(layer_ids, dtype=torch.long)
            out = []
            for seq in seqs:
                if common.numel() != seq.size(0):
                    raise ValueError(
                        f"layer_ids has {common.numel()} ids but sequence has M={seq.size(0)} layers."
                    )
                out.append(common.to(seq.device))
            return out

        if len(layer_ids) != len(seqs):
            raise ValueError("Per-example layer_ids list must match batch size.")

        out = []
        for i, (ids_i, seq) in enumerate(zip(layer_ids, seqs)):
            ids_i = (
                ids_i.long()
                if torch.is_tensor(ids_i)
                else torch.tensor(ids_i, dtype=torch.long)
            )
            if ids_i.numel() != seq.size(0):
                raise ValueError(
                    f"layer_ids[{i}] has {ids_i.numel()} ids but sequence has M={seq.size(0)} layers."
                )
            out.append(ids_i.to(seq.device))
        return out

    raise TypeError("Unsupported layer_ids format.")


class LocalTrajectoryBlock(nn.Module):
    """
    Strong per-layer temporal block:
    - multi-scale depthwise convs
    - content-dependent branch gating
    - gated channel mixing
    - squeeze-excitation
    - FFN
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        dilations: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 5,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.dilations = tuple(dilations)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dw_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * dil,
                    dilation=dil,
                    groups=d_model,
                )
                for dil in self.dilations
            ]
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.dilations)),
        )

        self.mix_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.mix_out = nn.Conv1d(d_model, d_model, kernel_size=1)

        se_hidden = max(32, d_model // 8)
        self.se = nn.Sequential(
            nn.Conv1d(d_model, se_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(se_hidden, d_model, kernel_size=1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        u = self.ln1(x)
        u_t = u.transpose(1, 2).contiguous()  # [B, D, S]

        ys = [conv(u_t) for conv in self.dw_convs]
        gates = torch.softmax(self.branch_gate(u), dim=-1)  # [B, S, n_branches]

        mix = 0.0
        for i, y in enumerate(ys):
            mix = mix + y * gates[..., i].unsqueeze(1)

        mix = self.mix_in(mix)
        a, g = mix.chunk(2, dim=1)
        mix = a * torch.sigmoid(g)
        mix = self.mix_out(mix)

        se = self.se(mix.mean(dim=-1, keepdim=True))
        mix = mix * se

        x = x + self.dropout(mix.transpose(1, 2).contiguous())
        x = x + self.ffn(self.ln2(x))
        return x


class LayerAxisBlock(nn.Module):
    """
    Self-attention over the layer dimension at each token position.
    Input shape is [S, M, D] where S acts as batch.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = FeedForward(d_model, mult=ffn_mult, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, M, D]
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(y)
        x = x + self.ffn(self.ln2(x))
        return x


class RelativeDistanceAttention(nn.Module):
    """
    Attention with ALiBi-style distance penalty based on actual positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        slopes = _get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if positions is not None:
            dist = (
                (positions.unsqueeze(-1) - positions.unsqueeze(-2))
                .abs()
                .to(scores.dtype)
            )
            bias = -self.slopes.to(scores.dtype).view(
                1, self.n_heads, 1, 1
            ) * torch.log1p(dist).unsqueeze(1)
            scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_drop(self.out(out))
        return out


class GlobalMixerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = RelativeDistanceAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.ffn = FeedForward(d_model=d_model, mult=ffn_mult, dropout=dropout)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions=positions)
        x = x + self.ffn(self.ln2(x))
        return x


class MultiLayerInputFusion(nn.Module):
    """
    Builds token-layer features from:
      - h
      - token delta
      - layer delta
      - mixed delta
      - scalar dynamics

    Input:  [M, S, D_in]
    Output: [M, S, d_model]
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm_h = nn.LayerNorm(d_in)
        self.norm_dt = nn.LayerNorm(d_in)
        self.norm_dl = nn.LayerNorm(d_in)
        self.norm_dtl = nn.LayerNorm(d_in)

        self.h_proj = nn.Linear(d_in, d_model)
        self.dt_proj = nn.Linear(d_in, d_model)
        self.dl_proj = nn.Linear(d_in, d_model)
        self.dtl_proj = nn.Linear(d_in, d_model)

        # scalars:
        # ||h||, ||dt||, ||dl||, ||dtl||, cos(prev token), cos(prev layer)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.gate = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 5),
        )

        self.out = nn.Sequential(
            nn.Linear(6 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [M, S, D]
        prev_t = torch.cat([h[:, :1], h[:, :-1]], dim=1)
        dt = h - prev_t

        prev_l = torch.cat([h[:1], h[:-1]], dim=0)
        dl = h - prev_l

        prev_dt_l = torch.cat([dt[:1], dt[:-1]], dim=0)
        dtl = dt - prev_dt_l

        cos_t = F.cosine_similarity(h, prev_t, dim=-1, eps=1e-6).unsqueeze(-1)
        cos_l = F.cosine_similarity(h, prev_l, dim=-1, eps=1e-6).unsqueeze(-1)

        scalars = torch.cat(
            [
                h.norm(dim=-1, keepdim=True),
                dt.norm(dim=-1, keepdim=True),
                dl.norm(dim=-1, keepdim=True),
                dtl.norm(dim=-1, keepdim=True),
                cos_t,
                cos_l,
            ],
            dim=-1,
        )

        ph = self.h_proj(self.norm_h(h))
        pdt = self.dt_proj(self.norm_dt(dt))
        pdl = self.dl_proj(self.norm_dl(dl))
        pdtl = self.dtl_proj(self.norm_dtl(dtl))
        ps = self.scalar_mlp(scalars)

        cat = torch.cat([ph, pdt, pdl, pdtl, ps], dim=-1)
        w = torch.softmax(self.gate(cat), dim=-1)

        parts = torch.stack([ph, pdt, pdl, pdtl, ps], dim=-2)  # [M, S, 5, d]
        fused = (parts * w.unsqueeze(-1)).sum(dim=-2)

        out = self.out(torch.cat([fused, ph, pdt, pdl, pdtl, ps], dim=-1))
        return out


class StrongMultiLayerHiddenTrajectoryEncoder(nn.Module):
    """
    Strong multi-layer hidden-state encoder.

    Input:
      - hidden: List[Tensor[M,S,D]] or Tensor[B,M,S,D]
      - lengths: optional Tensor[B]
      - layer_ids:
          * None
          * Tensor[M]
          * Tensor[B,M]
          * list[int]
          * list[tensor/list[int]]

    Output:
      - z_hid: [B, D_HID]
    """

    def __init__(
        self,
        D_model: int,
        D_HID: int = 1024,
        d_probe: int = 768,
        max_model_layers: int = 128,
        # per-layer temporal encoder
        local_layers: int = 6,
        local_kernel_size: int = 5,
        local_dilations: Sequence[int] = (1, 2, 4, 8),
        # tokenwise cross-layer fusion
        layer_fusion_layers: int = 3,
        layer_fusion_heads: int = 8,
        # summaries
        layer_summary_tokens: int = 2,
        chunk_size: int = 128,
        chunk_summary_tokens: int = 2,
        memory_tokens: int = 16,
        memory_layers: int = 2,
        # saliency selection
        recent_tokens: int = 96,
        min_global_salient: int = 32,
        max_global_salient: int = 256,
        salient_scale: float = 4.0,
        salient_per_chunk: int = 1,
        change_weight: float = 0.35,
        # global bank mixer
        global_layers: int = 4,
        global_heads: int = 12,
        # final descriptor readout
        final_queries: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_probe % layer_fusion_heads != 0:
            raise ValueError("d_probe must be divisible by layer_fusion_heads.")
        if d_probe % global_heads != 0:
            raise ValueError("d_probe must be divisible by global_heads.")

        self.D_model = D_model
        self.D_HID = D_HID
        self.d_probe = d_probe
        self.max_model_layers = int(max_model_layers)

        self.chunk_size = int(chunk_size)
        self.chunk_summary_tokens = int(chunk_summary_tokens)
        self.layer_summary_tokens = int(layer_summary_tokens)
        self.memory_tokens = int(memory_tokens)
        self.recent_tokens = int(recent_tokens)

        self.min_global_salient = int(min_global_salient)
        self.max_global_salient = int(max_global_salient)
        self.salient_scale = float(salient_scale)
        self.salient_per_chunk = int(salient_per_chunk)
        self.change_weight = float(change_weight)

        # embeddings
        self.layer_id_emb = nn.Embedding(self.max_model_layers, d_probe)
        self.bank_type_emb = nn.Embedding(
            5, d_probe
        )  # memory, chunk, salient, recent, layer_summary

        # A) multi-layer input fusion
        self.input_fusion = MultiLayerInputFusion(
            d_in=D_model, d_model=d_probe, dropout=dropout
        )

        # B) shared temporal encoder applied independently to each layer
        self.local_blocks = nn.ModuleList(
            [
                LocalTrajectoryBlock(
                    d_model=d_probe,
                    dropout=dropout,
                    dilations=local_dilations,
                    kernel_size=local_kernel_size,
                    ffn_mult=4,
                )
                for _ in range(local_layers)
            ]
        )

        # C) tokenwise cross-layer fusion
        self.layer_blocks = nn.ModuleList(
            [
                LayerAxisBlock(
                    d_model=d_probe,
                    n_heads=layer_fusion_heads,
                    dropout=dropout,
                    ffn_mult=4,
                )
                for _ in range(layer_fusion_layers)
            ]
        )

        # one shared fused token per time step
        self.layer_token_pool = QueryDecoder(
            d_model=d_probe,
            n_heads=layer_fusion_heads,
            num_queries=1,
            dropout=dropout,
            ffn_mult=4,
        )

        # D) saliency
        sal_in_dim = 3 * d_probe + 2
        chg_in_dim = 2 * d_probe + 2

        self.saliency_scorer = nn.Sequential(
            nn.LayerNorm(sal_in_dim),
            nn.Linear(sal_in_dim, 2 * d_probe),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_probe, 1),
        )

        self.change_scorer = nn.Sequential(
            nn.LayerNorm(chg_in_dim),
            nn.Linear(chg_in_dim, d_probe),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_probe, 1),
        )

        # E) chunk summaries + memory
        self.chunk_decoder = QueryDecoder(
            d_model=d_probe,
            n_heads=global_heads,
            num_queries=chunk_summary_tokens,
            dropout=dropout,
            ffn_mult=4,
        )

        self.memory_rnn = nn.GRU(
            input_size=d_probe,
            hidden_size=d_probe,
            num_layers=memory_layers,
            dropout=(dropout if memory_layers > 1 else 0.0),
            batch_first=True,
        )

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(d_probe),
            nn.Linear(d_probe, d_probe),
            nn.GELU(),
        )

        # per-layer summary tokens
        self.layer_summary_decoder = QueryDecoder(
            d_model=d_probe,
            n_heads=global_heads,
            num_queries=layer_summary_tokens,
            dropout=dropout,
            ffn_mult=4,
        )

        # F) global bank mixer
        self.global_blocks = nn.ModuleList(
            [
                GlobalMixerBlock(
                    d_model=d_probe,
                    n_heads=global_heads,
                    dropout=dropout,
                    ffn_mult=4,
                )
                for _ in range(global_layers)
            ]
        )

        # G) final descriptor readout
        self.final_decoder = QueryDecoder(
            d_model=d_probe,
            n_heads=global_heads,
            num_queries=final_queries,
            dropout=dropout,
            ffn_mult=4,
        )

        self.final_head = QueryProjectionHead(
            d_model=d_probe,
            num_queries=final_queries,
            out_dim=D_HID,
            dropout=dropout,
        )

    # -----------------------------------------------------
    # utilities
    # -----------------------------------------------------

    def _validate_layer_ids(self, layer_ids: torch.Tensor) -> torch.Tensor:
        layer_ids = layer_ids.long()
        if layer_ids.numel() == 0:
            raise ValueError("layer_ids is empty.")
        if layer_ids.min().item() < 0:
            raise ValueError("layer_ids must be non-negative.")
        if layer_ids.max().item() >= self.max_model_layers:
            raise ValueError(
                f"layer_ids max={int(layer_ids.max().item())} exceeds max_model_layers={self.max_model_layers}."
            )
        return layer_ids

    def _temporal_encode_layers(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, S, d]
        for blk in self.local_blocks:
            x = blk(x)
        return x

    def _tokenwise_layer_fusion(
        self, x_layers: torch.Tensor, layer_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        x_layers: [M, S, d]
        Returns:
          shared_stream: [S, d]
        """
        layer_ids = self._validate_layer_ids(layer_ids)
        u = x_layers.permute(1, 0, 2).contiguous()  # [S, M, d]
        u = u + self.layer_id_emb(layer_ids).unsqueeze(0)

        for blk in self.layer_blocks:
            u = blk(u)

        pooled = self.layer_token_pool(u).squeeze(1)  # [S, d]
        return pooled

    def _layer_agreement_features(self, x_layers: torch.Tensor):
        """
        x_layers: [M, S, d]
        """
        M, S, _D = x_layers.shape
        layer_std = x_layers.std(dim=0, unbiased=False)  # [S, d]

        if M > 1:
            layer_diff = x_layers[1:] - x_layers[:-1]  # [M-1, S, d]
            mean_layer_delta_norm = (
                layer_diff.norm(dim=-1).mean(dim=0).unsqueeze(-1)
            )  # [S,1]

            adj_cos = F.cosine_similarity(x_layers[1:], x_layers[:-1], dim=-1, eps=1e-6)
            mean_adj_cos = adj_cos.mean(dim=0).unsqueeze(-1)  # [S,1]
        else:
            mean_layer_delta_norm = torch.zeros(
                S, 1, device=x_layers.device, dtype=x_layers.dtype
            )
            mean_adj_cos = torch.ones(
                S, 1, device=x_layers.device, dtype=x_layers.dtype
            )

        return layer_std, mean_layer_delta_norm, mean_adj_cos

    def _select_salient_indices(
        self, shared_stream: torch.Tensor, refined_layers: torch.Tensor
    ):
        """
        shared_stream: [S, d]
        refined_layers: [M, S, d]
        """
        S = shared_stream.size(0)

        prev = torch.cat([shared_stream[:1], shared_stream[:-1]], dim=0)
        dx = shared_stream - prev

        layer_std, mean_layer_delta_norm, mean_adj_cos = self._layer_agreement_features(
            refined_layers
        )

        sal_in = torch.cat(
            [
                shared_stream,
                dx.abs(),
                layer_std,
                mean_layer_delta_norm,
                1.0 - mean_adj_cos,
            ],
            dim=-1,
        )

        raw_sal = self.saliency_scorer(sal_in).squeeze(-1)

        chg_in = torch.cat(
            [
                dx.abs(),
                layer_std,
                mean_layer_delta_norm,
                1.0 - mean_adj_cos,
            ],
            dim=-1,
        )

        change = self.change_scorer(chg_in).squeeze(-1)

        score = raw_sal + self.change_weight * change

        k_global = round(self.salient_scale * math.sqrt(float(S)))
        k_global = max(self.min_global_salient, k_global)
        k_global = min(self.max_global_salient, k_global)
        k_global = min(S, max(1, k_global))

        global_idx = torch.topk(score, k=k_global, dim=0).indices

        per_chunk_idx = []
        for st in range(0, S, self.chunk_size):
            ed = min(st + self.chunk_size, S)
            local_k = min(self.salient_per_chunk, ed - st)
            local_idx = st + torch.topk(score[st:ed], k=local_k, dim=0).indices
            per_chunk_idx.append(local_idx)

        salient_idx = torch.cat([global_idx] + per_chunk_idx, dim=0).unique(sorted=True)

        recent_start = max(0, S - self.recent_tokens)
        recent_idx = torch.arange(recent_start, S, device=shared_stream.device)

        return salient_idx, recent_idx

    def _summarize_chunks(self, shared_stream: torch.Tensor):
        """
        shared_stream: [S, d]
        """
        S, _D = shared_stream.shape
        chunk_tokens = []
        chunk_positions = []
        chunk_means = []
        chunk_end_positions = []

        for st in range(0, S, self.chunk_size):
            ed = min(st + self.chunk_size, S)
            chunk = shared_stream[st:ed].unsqueeze(0)  # [1, L, d]
            summary = self.chunk_decoder(chunk).squeeze(0)  # [Q, d]

            chunk_tokens.append(summary)
            chunk_positions.append(
                torch.full(
                    (summary.size(0),),
                    fill_value=0.5 * (st + ed - 1),
                    device=shared_stream.device,
                    dtype=shared_stream.dtype,
                )
            )
            chunk_means.append(summary.mean(dim=0))
            chunk_end_positions.append(float(ed - 1))

        chunk_tokens = torch.cat(chunk_tokens, dim=0)
        chunk_positions = torch.cat(chunk_positions, dim=0)
        chunk_means = torch.stack(chunk_means, dim=0).unsqueeze(0)

        mem_seq, _ = self.memory_rnn(chunk_means)
        keep = min(self.memory_tokens, mem_seq.size(1))
        memory_tokens = self.memory_proj(mem_seq[:, -keep:, :]).squeeze(0)
        memory_positions = torch.tensor(
            chunk_end_positions[-keep:],
            device=shared_stream.device,
            dtype=shared_stream.dtype,
        )

        return chunk_tokens, chunk_positions, memory_tokens, memory_positions

    def _layer_summary_tokens(self, x_layers: torch.Tensor, layer_ids: torch.Tensor):
        """
        x_layers: [M, S, d]
        """
        layer_ids = self._validate_layer_ids(layer_ids)
        M, S, D = x_layers.shape

        summaries = self.layer_summary_decoder(x_layers)  # [M, Q, d]
        summaries = summaries + self.layer_id_emb(layer_ids).unsqueeze(1)

        tokens = summaries.reshape(M * self.layer_summary_tokens, D)
        positions = torch.full(
            (M * self.layer_summary_tokens,),
            fill_value=0.5 * (S - 1),
            device=x_layers.device,
            dtype=x_layers.dtype,
        )
        return tokens, positions

    def _build_bank(
        self,
        shared_stream: torch.Tensor,
        refined_layers: torch.Tensor,
        layer_ids: torch.Tensor,
    ):
        salient_idx, recent_idx = self._select_salient_indices(
            shared_stream, refined_layers
        )

        salient_tokens = shared_stream[salient_idx]
        recent_tokens = shared_stream[recent_idx]

        salient_pos = salient_idx.to(dtype=shared_stream.dtype)
        recent_pos = recent_idx.to(dtype=shared_stream.dtype)

        chunk_tokens, chunk_positions, memory_tokens, memory_positions = (
            self._summarize_chunks(shared_stream)
        )
        layer_summary_tokens, layer_summary_positions = self._layer_summary_tokens(
            refined_layers, layer_ids
        )

        bank_tokens = torch.cat(
            [
                memory_tokens,
                chunk_tokens,
                salient_tokens,
                recent_tokens,
                layer_summary_tokens,
            ],
            dim=0,
        )

        bank_pos = torch.cat(
            [
                memory_positions,
                chunk_positions,
                salient_pos,
                recent_pos,
                layer_summary_positions,
            ],
            dim=0,
        )

        bank_type = torch.cat(
            [
                torch.zeros(
                    memory_tokens.size(0), device=shared_stream.device, dtype=torch.long
                ),
                torch.ones(
                    chunk_tokens.size(0), device=shared_stream.device, dtype=torch.long
                ),
                torch.full(
                    (salient_tokens.size(0),),
                    2,
                    device=shared_stream.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (recent_tokens.size(0),),
                    3,
                    device=shared_stream.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (layer_summary_tokens.size(0),),
                    4,
                    device=shared_stream.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )

        bank_tokens = bank_tokens + self.bank_type_emb(bank_type)

        order = torch.argsort(bank_pos)
        bank_tokens = bank_tokens[order]
        bank_pos = bank_pos[order]
        return bank_tokens, bank_pos

    def _encode_one(self, h: torch.Tensor, layer_ids: torch.Tensor) -> torch.Tensor:
        """
        h: [M, S, D_model]
        layer_ids: [M]
        """
        if h.dim() != 3:
            raise ValueError(
                f"Each sequence must have shape [M, S, D_model], got {tuple(h.shape)}"
            )

        _M, _S, D = h.shape
        if D != self.D_model:
            raise ValueError(f"Expected D_model={self.D_model}, got {D}")

        layer_ids = self._validate_layer_ids(layer_ids.to(h.device))

        # A) input fusion
        x = self.input_fusion(h)  # [M, S, d]
        x = x + self.layer_id_emb(layer_ids).unsqueeze(1)

        # B) per-layer temporal encoding
        x = self._temporal_encode_layers(x)  # [M, S, d]

        # C) tokenwise cross-layer fusion
        shared_stream = self._tokenwise_layer_fusion(x, layer_ids)  # [S, d]

        # D/E/F) build global bank and mix
        bank_tokens, bank_pos = self._build_bank(shared_stream, x, layer_ids)
        bank = bank_tokens.unsqueeze(0)
        bank_pos = bank_pos.unsqueeze(0)

        for blk in self.global_blocks:
            bank = blk(bank, positions=bank_pos)

        # G) final descriptor
        q = self.final_decoder(bank)
        z_hid = self.final_head(q).squeeze(0)  # [D_HID]
        return z_hid

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor | list[torch.Tensor],
        lengths: torch.Tensor | None = None,
        layer_ids: torch.Tensor
        | Sequence[int]
        | list[torch.Tensor | Sequence[int]]
        | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            z_hid: [B, D_HID]
        """
        seqs = _as_list_of_multilayer_sequences(hidden, lengths=lengths)
        ids_list = _as_list_of_layer_ids(layer_ids, seqs)

        outs = [self._encode_one(seq, ids_i) for seq, ids_i in zip(seqs, ids_list)]
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_prefixes(
        self,
        hidden_one: torch.Tensor,
        prefix_ends: Sequence[int],
        layer_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """
        hidden_one: [M, S, D_model]
        Returns:
            z_hid_prefixes: [P, D_HID]
        """
        if hidden_one.dim() != 3:
            raise ValueError("hidden_one must have shape [M, S, D_model].")

        if layer_ids is None:
            ids = torch.arange(
                hidden_one.size(0), device=hidden_one.device, dtype=torch.long
            )
        else:
            ids = (
                layer_ids
                if torch.is_tensor(layer_ids)
                else torch.tensor(layer_ids, dtype=torch.long, device=hidden_one.device)
            )

        prefix_outs = []
        for end in prefix_ends:
            end = int(max(1, min(end, hidden_one.size(1))))
            prefix_hidden = hidden_one[:, :end, :]
            prefix_outs.append(self.forward([prefix_hidden], layer_ids=[ids])[0])

        return torch.stack(prefix_outs, dim=0)


def _as_list_of_sequences(
    hidden: torch.Tensor | list[torch.Tensor],
    lengths: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Preferred:
        hidden = [Tensor[S_i, D], Tensor[S_j, D], ...]

    Also supports:
        hidden = Tensor[B, S, D] with optional lengths[B]
    """
    if isinstance(hidden, list):
        out = []
        for i, x in enumerate(hidden):
            if not torch.is_tensor(x):
                raise TypeError(f"hidden[{i}] must be a torch.Tensor, got {type(x)}")
            if x.dim() != 2:
                raise ValueError(
                    f"hidden[{i}] must have shape [S, D], got {tuple(x.shape)}"
                )
            out.append(x)
        return out

    if not torch.is_tensor(hidden):
        raise TypeError("hidden must be a Tensor[B,S,D] or List[Tensor[S_i,D]].")

    if hidden.dim() != 3:
        raise ValueError(
            f"Tensor input must have shape [B, S, D], got {tuple(hidden.shape)}"
        )

    B, S, _ = hidden.shape
    if lengths is None:
        return [hidden[b] for b in range(B)]

    if lengths.dim() != 1 or lengths.numel() != B:
        raise ValueError(f"lengths must have shape [B], got {tuple(lengths.shape)}")

    seqs = []
    for b in range(B):
        L = int(lengths[b].item())
        L = max(1, min(L, S))
        seqs.append(hidden[b, :L])
    return seqs


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = x.chunk(2, dim=-1)
        return a * F.gelu(g)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if mult <= 0:
            raise ValueError("mult must be > 0")

        inner = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, inner * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryProjectionHead(nn.Module):
    """
    Turns learned query outputs [B, Q, D] into [B, out_dim].
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int,
        out_dim: int,
        hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model <= 0 or num_queries <= 0 or out_dim <= 0:
            raise ValueError("d_model, num_queries, out_dim must be > 0")

        feat_dim = num_queries * d_model + 2 * d_model
        hidden_dim = hidden_mult * d_model
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.dim() != 3:
            raise ValueError(f"q must have shape [B, Q, D], got {tuple(q.shape)}")
        feat = torch.cat([q.flatten(1), q.mean(dim=1), q.max(dim=1).values], dim=-1)
        return self.net(feat)


class ContentProjection(nn.Module):
    """
    Strong content-only token featurizer using h_t only.
    Input:  [S, D_model]
    Output: [S, d_probe]
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if d_in <= 0 or d_model <= 0:
            raise ValueError("d_in and d_model must be > 0")

        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 2 * d_model),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"h must have shape [S, D_in], got {tuple(h.shape)}")
        return self.net(h)


class LocalContentBlock(nn.Module):
    """
    Multi-scale local content mixer:
    - multi-scale depthwise convs
    - content-dependent branch gating
    - pointwise GLU
    - FFN
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        dilations: Sequence[int] = (1, 2, 4),
        kernel_size: int = 5,
        ffn_mult: int = 2,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if len(dilations) == 0:
            raise ValueError("dilations must be non-empty")

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dw_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * int(d),
                    dilation=int(d),
                    groups=d_model,
                )
                for d in dilations
            ]
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(dilations)),
        )

        self.mix_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.mix_out = nn.Conv1d(d_model, d_model, kernel_size=1)

        se_hidden = max(32, d_model // 8)
        self.se = nn.Sequential(
            nn.Conv1d(d_model, se_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(se_hidden, d_model, kernel_size=1),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model=d_model, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, S, D], got {tuple(x.shape)}")

        u = self.ln1(x)
        u_t = u.transpose(1, 2).contiguous()  # [B, D, S]

        ys = [conv(u_t) for conv in self.dw_convs]  # each [B, D, S]
        gates = torch.softmax(self.branch_gate(u), dim=-1)  # [B, S, n_branches]

        mix = 0.0
        for i, y in enumerate(ys):
            mix = mix + y * gates[..., i].unsqueeze(1)

        mix = self.mix_in(mix)
        a, g = mix.chunk(2, dim=1)
        mix = a * torch.sigmoid(g)
        mix = self.mix_out(mix)

        se = self.se(mix.mean(dim=-1, keepdim=True))
        mix = mix * se

        x = x + self.drop(mix.transpose(1, 2).contiguous())
        x = x + self.ffn(self.ln2(x))
        return x


class LatentCrossBlock(nn.Module):
    """
    Perceiver-style latent update:
    - latents cross-attend to context
    - latent self-attn
    - latent FFN
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 2
    ):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model and n_heads must be > 0")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.ln_lat_cross = nn.LayerNorm(d_model)
        self.ln_ctx = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_lat_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, mult=ffn_mult, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(
                f"latents must have shape [B, L, D], got {tuple(latents.shape)}"
            )
        if context.dim() != 3:
            raise ValueError(
                f"context must have shape [B, N, D], got {tuple(context.shape)}"
            )

        q = self.ln_lat_cross(latents)
        ctx = self.ln_ctx(context)
        cross_out, _ = self.cross_attn(q, ctx, ctx, need_weights=False)
        latents = latents + self.drop(cross_out)

        s = self.ln_lat_self(latents)
        self_out, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + self.drop(self_out)

        latents = latents + self.ffn(self.ln_ff(latents))
        return latents


class QueryDecoder(nn.Module):
    """
    Learned queries cross-attend into context, then self-refine.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_queries: int,
        dropout: float = 0.1,
        ffn_mult: int = 2,
    ):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or num_queries <= 0:
            raise ValueError("d_model, n_heads, num_queries must be > 0")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.num_queries = num_queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, d_model) / math.sqrt(d_model)
        )

        self.ln_q_cross = nn.LayerNorm(d_model)
        self.ln_ctx = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_q_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, mult=ffn_mult, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.dim() != 3:
            raise ValueError(
                f"context must have shape [B, N, D], got {tuple(context.shape)}"
            )

        B = context.size(0)
        q = self.queries.expand(B, -1, -1)

        ctx = self.ln_ctx(context)
        q1 = self.ln_q_cross(q)
        cross_out, _ = self.cross_attn(q1, ctx, ctx, need_weights=False)
        q = q + self.drop(cross_out)

        q2 = self.ln_q_self(q)
        self_out, _ = self.self_attn(q2, q2, q2, need_weights=False)
        q = q + self.drop(self_out)

        q = q + self.ffn(self.ln_ff(q))
        return q


class StrongContentOnlyHiddenEncoder(nn.Module):
    """
    Strong content-only hidden-state encoder.

    Design:
      h_t
      -> content projection
      -> local content blocks
      -> chunk compression + recent token retention
      -> latent bottleneck
      -> learned query readout
      -> z_hid

    Input:
      - List[Tensor[S_i, D_model]]
      - or Tensor[B, S, D_model] with optional lengths

    Output:
      - z_hid: [B, D_HID]
    """

    def __init__(
        self,
        D_model: int,
        D_HID: int = 256,
        d_probe: int = 384,
        # local sequence encoder
        local_layers: int = 3,
        local_kernel_size: int = 5,
        local_dilations: Sequence[int] = (1, 2, 4),
        # chunk compression
        chunk_size: int = 128,
        keep_recent_tokens: int = 48,
        # latent bottleneck
        latent_tokens: int = 32,
        latent_layers: int = 2,
        latent_heads: int = 8,
        # readout
        readout_queries: int = 4,
        readout_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if D_model <= 0 or D_HID <= 0 or d_probe <= 0:
            raise ValueError("D_model, D_HID, d_probe must be > 0")
        if local_layers <= 0:
            raise ValueError("local_layers must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if keep_recent_tokens < 0:
            raise ValueError("keep_recent_tokens must be >= 0")
        if latent_tokens <= 0 or latent_layers <= 0:
            raise ValueError("latent_tokens and latent_layers must be > 0")
        if readout_queries <= 0:
            raise ValueError("readout_queries must be > 0")
        if d_probe % latent_heads != 0:
            raise ValueError(
                f"d_probe={d_probe} must be divisible by latent_heads={latent_heads}"
            )
        if d_probe % readout_heads != 0:
            raise ValueError(
                f"d_probe={d_probe} must be divisible by readout_heads={readout_heads}"
            )

        self.D_model = int(D_model)
        self.D_HID = int(D_HID)
        self.d_probe = int(d_probe)
        self.chunk_size = int(chunk_size)
        self.keep_recent_tokens = int(keep_recent_tokens)

        # 1) content projection
        self.input_proj = ContentProjection(
            d_in=D_model, d_model=d_probe, dropout=dropout
        )

        # 2) local sequence backbone
        self.local_blocks = nn.ModuleList(
            [
                LocalContentBlock(
                    d_model=d_probe,
                    dropout=dropout,
                    dilations=local_dilations,
                    kernel_size=local_kernel_size,
                    ffn_mult=2,
                )
                for _ in range(local_layers)
            ]
        )

        # 3) chunk compressor
        self.chunk_proj = nn.Sequential(
            nn.LayerNorm(3 * d_probe),
            nn.Linear(3 * d_probe, 2 * d_probe),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_probe, d_probe),
        )

        # distinguish chunk summaries vs recent raw tokens
        self.context_type_emb = nn.Embedding(2, d_probe)  # 0=chunk, 1=recent

        # 4) latent bottleneck
        self.latents = nn.Parameter(
            torch.randn(1, latent_tokens, d_probe) / math.sqrt(d_probe)
        )
        self.latent_blocks = nn.ModuleList(
            [
                LatentCrossBlock(
                    d_model=d_probe,
                    n_heads=latent_heads,
                    dropout=dropout,
                    ffn_mult=2,
                )
                for _ in range(latent_layers)
            ]
        )

        # 5) learned readout
        self.readout = QueryDecoder(
            d_model=d_probe,
            n_heads=readout_heads,
            num_queries=readout_queries,
            dropout=dropout,
            ffn_mult=2,
        )
        self.out_head = QueryProjectionHead(
            d_model=d_probe,
            num_queries=readout_queries,
            out_dim=D_HID,
            dropout=dropout,
        )

    # -----------------------------------------------------
    # internal utilities
    # -----------------------------------------------------

    def _local_encode(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"h must have shape [S, D_model], got {tuple(h.shape)}")
        x = self.input_proj(h).unsqueeze(0)  # [1, S, d_probe]
        for blk in self.local_blocks:
            x = blk(x)
        return x.squeeze(0)  # [S, d_probe]

    def _compress_chunks(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [S, d_probe]
        Returns chunk summary tokens: [C, d_probe]
        """
        if x.dim() != 2:
            raise ValueError(f"x must have shape [S, d_probe], got {tuple(x.shape)}")

        S, _D = x.shape
        chunk_tokens = []

        for st in range(0, S, self.chunk_size):
            ed = min(st + self.chunk_size, S)
            c = x[st:ed]  # [L, D]

            c_mean = c.mean(dim=0)
            c_max = c.max(dim=0).values
            c_last = c[-1]

            feat = torch.cat([c_mean, c_max, c_last], dim=-1)  # [3D]
            tok = self.chunk_proj(feat)
            chunk_tokens.append(tok)

        return torch.stack(chunk_tokens, dim=0)  # [C, D]

    def _build_context(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [S, d_probe]
        Returns context tokens [N, d_probe]
        """
        chunk_tokens = self._compress_chunks(x)
        chunk_types = torch.zeros(
            chunk_tokens.size(0), device=x.device, dtype=torch.long
        )
        chunk_tokens = chunk_tokens + self.context_type_emb(chunk_types)

        parts = [chunk_tokens]

        if self.keep_recent_tokens > 0:
            recent = x[-min(self.keep_recent_tokens, x.size(0)) :]  # [R, D]
            recent_types = torch.ones(recent.size(0), device=x.device, dtype=torch.long)
            recent = recent + self.context_type_emb(recent_types)
            parts.append(recent)

        context = torch.cat(parts, dim=0)  # [N, D]
        return {
            "context_tokens": context,
            "chunk_tokens": chunk_tokens,
        }

    def _encode_one(self, h: torch.Tensor, return_debug: bool = False):
        if not torch.is_tensor(h):
            raise TypeError("Each sequence must be a torch.Tensor.")
        if h.dim() != 2:
            raise ValueError(
                f"Each sequence must have shape [S, D_model], got {tuple(h.shape)}"
            )
        if h.size(0) < 1:
            raise ValueError("Sequence length must be at least 1.")
        if h.size(1) != self.D_model:
            raise ValueError(f"Expected hidden size {self.D_model}, got {h.size(1)}")

        # local token encoding
        x = self._local_encode(h)  # [S, d_probe]

        # compressed context
        ctx_info = self._build_context(x)
        context = ctx_info["context_tokens"].unsqueeze(0)  # [1, N, d_probe]

        # latent bottleneck
        latents = self.latents.expand(1, -1, -1)
        for blk in self.latent_blocks:
            latents = blk(latents, context)

        # readout
        q = self.readout(latents)
        z_hid = self.out_head(q).squeeze(0)  # [D_HID]

        if return_debug:
            return {
                "z_hid": z_hid,
                "debug": {
                    "local_tokens": x,
                    "context_tokens": context.squeeze(0),
                    "chunk_tokens": ctx_info["chunk_tokens"],
                    "latents": latents.squeeze(0),
                },
            }

        return z_hid

    # -----------------------------------------------------
    # public API
    # -----------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor | list[torch.Tensor],
        lengths: torch.Tensor | None = None,
        return_debug: bool = False,
    ):
        seqs = _as_list_of_sequences(hidden, lengths=lengths)

        outs = [self._encode_one(seq, return_debug=return_debug) for seq in seqs]

        if return_debug:
            return {
                "z_hid": torch.stack([o["z_hid"] for o in outs], dim=0),
                "debug": [o["debug"] for o in outs],
            }

        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_prefixes(
        self,
        hidden_one: torch.Tensor,
        prefix_ends: Sequence[int],
        return_debug: bool = False,
    ):
        if not torch.is_tensor(hidden_one):
            raise TypeError("hidden_one must be a torch.Tensor.")
        if hidden_one.dim() != 2:
            raise ValueError("hidden_one must have shape [S, D_model].")
        if hidden_one.size(0) < 1:
            raise ValueError("hidden_one must contain at least one token.")
        if hidden_one.size(1) != self.D_model:
            raise ValueError(
                f"Expected hidden size {self.D_model}, got {hidden_one.size(1)}"
            )

        prefix_outs = []
        for end in prefix_ends:
            end = int(max(1, min(end, hidden_one.size(0))))
            prefix_outs.append(
                self.forward([hidden_one[:end]], return_debug=return_debug)
            )

        if return_debug:
            return {
                "z_hid": torch.cat([o["z_hid"] for o in prefix_outs], dim=0),
                "debug": [o["debug"][0] for o in prefix_outs],
            }

        return torch.cat(prefix_outs, dim=0)
