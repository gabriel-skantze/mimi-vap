import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import Tensor
from einops.layers.torch import Rearrange
from typing import Dict, Optional, Tuple, Mapping


@dataclass
class KVCache:
    """Bounded projected key/value state for incremental attention."""

    key: Optional[Tensor] = None
    value: Optional[Tensor] = None
    next_position: int = 0

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[-2]


@dataclass
class TransformerLayerCache:
    self_attention: KVCache = field(default_factory=KVCache)
    cross_attention: Optional[KVCache] = None


@dataclass
class TransformerStereoCache:
    batch_size: int
    max_context: int
    channel_layers: list[TransformerLayerCache]
    stereo_layers: list[TransformerLayerCache]


def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        attention_scale: str = "model",
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.attention_scale = attention_scale.lower()
        if self.attention_scale not in {"model", "head"}:
            raise ValueError(
                "attention_scale must be 'model' or 'head', got "
                f"{attention_scale!r}"
            )

        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=bias)
        if self.attention_scale == "head":
            # Persist the corrected choice so compact checkpoints can be
            # distinguished from the repository's legacy model-dim scaling.
            self.register_buffer(
                "scale",
                torch.tensor(1.0 / math.sqrt(self.head_dim)),
                persistent=True,
            )
        else:
            self.scale = 1.0 / math.sqrt(dim)

    def get_scores(self, q: Tensor, k: Tensor) -> Tensor:
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    @staticmethod
    def prepare_causal_mask(
        T, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> Tensor:
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype)).view(
            1, 1, T, T
        )
        mask.requires_grad_(False)
        return mask

    def mask_scores(self, qk: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        T = qk.size(-1)
        if mask is None:
            mask = MultiHeadAttention.prepare_causal_mask(
                T, device=qk.device, dtype=qk.dtype
            )
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def position_queries_and_keys(
        self,
        q: Tensor,
        k: Tensor,
        q_positions: Optional[Tensor] = None,
        k_positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return q, k

    def incremental_score_bias(
        self, key_positions: Tensor, dtype: torch.dtype
    ) -> Optional[Tensor]:
        return None

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)
        q, k = self.position_queries_and_keys(q, k)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

    def forward_incremental(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        cache: KVCache,
        max_context: int,
    ) -> Tuple[Tensor, Tensor]:
        """Attend new queries to a bounded cache plus the new keys and values.

        Positions remain monotonic after old cache entries are evicted.  This is
        required for RoPE and also makes chunked calls equivalent to framewise
        calls. ``max_context`` includes the current position.
        """
        if max_context <= 0:
            raise ValueError(f"max_context must be positive, got {max_context}")
        if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
            raise ValueError("Incremental attention expects [batch, time, dim]")
        if Q.shape[:2] != K.shape[:2] or K.shape != V.shape:
            raise ValueError(
                "Incremental attention requires aligned Q/K/V chunks, got "
                f"{tuple(Q.shape)}, {tuple(K.shape)}, {tuple(V.shape)}"
            )

        batch_size, chunk_frames, _ = Q.shape
        q = self.unstack_heads(self.query(Q))
        k = self.unstack_heads(self.key(K))
        v = self.unstack_heads(self.value(V))

        start = cache.next_position
        q_positions = torch.arange(
            start, start + chunk_frames, device=Q.device, dtype=torch.long
        )
        k_positions_new = q_positions
        q, k = self.position_queries_and_keys(
            q, k, q_positions=q_positions, k_positions=k_positions_new
        )

        if cache.key is None:
            key = k
            value = v
            key_positions = k_positions_new
        else:
            if cache.value is None:
                raise RuntimeError("KV cache has keys but no values")
            if cache.key.shape[0] != batch_size:
                raise ValueError(
                    f"KV cache batch size {cache.key.shape[0]} != {batch_size}"
                )
            old_start = start - cache.length
            old_positions = torch.arange(
                old_start, start, device=Q.device, dtype=torch.long
            )
            key = torch.cat((cache.key, k), dim=-2)
            value = torch.cat((cache.value, v), dim=-2)
            key_positions = torch.cat((old_positions, k_positions_new))

        scores = self.get_scores(q, key) * self.scale
        delta = q_positions[:, None] - key_positions[None, :]
        causal_window = (delta >= 0) & (delta < max_context)
        scores = scores.masked_fill(
            ~causal_window.view(1, 1, chunk_frames, -1), float("-inf")
        )
        bias = self.incremental_score_bias(key_positions, scores.dtype)
        if bias is not None:
            scores = scores + bias
        att = F.softmax(scores, dim=-1)
        y = self.attn_drop(att) @ value
        y = self.stack_heads(y)
        y = self.resid_drop(self.proj(y))

        cache.key = key[..., -max_context:, :]
        cache.value = value[..., -max_context:, :]
        cache.next_position += chunk_frames
        return y, att


class MultiHeadAttentionAlibi(MultiHeadAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        attention_scale: str = "model",
    ) -> None:
        super().__init__(dim, num_heads, dropout, bias, attention_scale)
        # self.m = torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))
        self.register_parameter(
            "m",
            nn.Parameter(torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))),
        )
        self.m.requires_grad_(False)
        self.mask = None

    @staticmethod
    def get_slopes(n: int) -> list[float]:
        """
        * aLiBi slopes for heads.
        * m in Figure 3.
        * Source:
            - https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742

        Comments:

        In the paper, we only train models that have 2^a heads for some a. This function has
        some good properties that only occur when the input is a power of 2.
        To maintain that even closest_power_of_2 = 2**math.floor(math.log2(n))
        when the number of heads is not a power of 2, we use this workaround.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + MultiHeadAttentionAlibi.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )
        return slopes

    @staticmethod
    def get_relative_bias_matrix(
        n: int, num_heads: int, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Relative Bias matrix for aLiBi embeddings"""
        return (
            torch.arange(n, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(1, num_heads, -1)
        )

    def get_alibi_mask(self, T: int, device="cpu", dtype=torch.float32):
        rel_bias_mat = MultiHeadAttentionAlibi.get_relative_bias_matrix(
            T, self.num_heads, device, dtype
        )
        alibi = rel_bias_mat * self.m.unsqueeze(0).unsqueeze(-1).to(device)

        # Causal mask (standard GPT pask)
        # lower triangle = 1
        # upper triangle = 0
        mask = MultiHeadAttention.prepare_causal_mask(T, device, dtype)  # (1, 1, T, T)
        # Repeat to get a mask for each head
        mask = mask.repeat(1, self.num_heads, 1, 1)  # (1, num_heads, T, T)
        # fill "future" information with negative infinity
        mask.masked_fill_(mask == 0, float("-inf"))

        # Add causality mask to alibi  (1, num_heads, T, T)
        alibi = alibi.unsqueeze(-2) + mask
        alibi.requires_grad_(False)  # this should not be trained
        return alibi

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            if self.mask is None or self.mask.shape[-1] < T:
                mask = self.get_alibi_mask(T, device=qk.device, dtype=qk.dtype)
                self.mask = mask
            else:
                mask = self.mask[..., :T, :T]

        # add aLiBi-mask to qk (see Figure 3.)
        # Addition/translation does not effect softmax (over each row)
        # mentioned in the original representation
        qk = qk + mask.to(qk.device)
        return qk

    def incremental_score_bias(
        self, key_positions: Tensor, dtype: torch.dtype
    ) -> Tensor:
        # ALiBi is translation invariant within a softmax row. Rebasing avoids
        # loss of precision as the absolute stream position grows indefinitely.
        positions = (key_positions - key_positions[0]).to(dtype=dtype).view(
            1, 1, 1, -1
        )
        slopes = self.m.to(device=key_positions.device, dtype=dtype).view(
            1, self.num_heads, 1, 1
        )
        return positions * slopes


class MultiHeadAttentionRoPE(MultiHeadAttention):
    """Causal attention with interleaved rotary position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        theta: float = 10_000.0,
        attention_scale: str = "model",
    ) -> None:
        super().__init__(dim, num_heads, dropout, bias, attention_scale)
        head_dim = dim // num_heads
        if head_dim % 2:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def _rotate(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        frames = x.shape[-2]
        if positions is None:
            positions = torch.arange(frames, device=x.device, dtype=torch.float32)
        else:
            positions = positions.to(device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=x.device))
        cos = angles.cos().to(dtype=x.dtype).view(1, 1, frames, -1)
        sin = angles.sin().to(dtype=x.dtype).view(1, 1, frames, -1)
        pairs = x.unflatten(-1, (-1, 2))
        even, odd = pairs.unbind(dim=-1)
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def position_queries_and_keys(
        self,
        q: Tensor,
        k: Tensor,
        q_positions: Optional[Tensor] = None,
        k_positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self._rotate(q, q_positions), self._rotate(k, k_positions)


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
    Inspiration: https://nn.labml.ai/transformers/models.html
    AliBI Attention: https://ofir.io/train_short_test_long.pdf
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        cross_attention: bool = False,
        positional_encoding: str = "alibi",
        attention_scale: str = "model",
        normalize_cross_source: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.positional_encoding = positional_encoding.lower()
        self.attention_scale = attention_scale.lower()
        self.normalize_cross_source = normalize_cross_source
        attention_class = {
            "alibi": MultiHeadAttentionAlibi,
            "rope": MultiHeadAttentionRoPE,
        }.get(self.positional_encoding)
        if attention_class is None:
            raise ValueError(
                "positional_encoding must be 'alibi' or 'rope', got "
                f"{positional_encoding!r}"
            )

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.mha = attention_class(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_scale=self.attention_scale,
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

        if cross_attention:
            self.ln_src_attn = nn.LayerNorm(dim)
            self.ln_cross_src = (
                nn.LayerNorm(dim) if normalize_cross_source else nn.Identity()
            )
            self.mha_cross = attention_class(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_scale=self.attention_scale,
            )

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
        """

        # Self-attention
        z = self.ln_self_attn(x)
        self_attn, self_attn_weights = self.mha(Q=z, K=z, V=z, mask=mask)

        # Residual
        x = x + self.dropout(self_attn)

        # Cross-attention
        cross_attn_weights = None
        if self.cross_attention and src is not None:
            z = self.ln_src_attn(x)
            src = self.ln_cross_src(src)
            cross_attn, cross_attn_weights = self.mha_cross(
                Q=z, K=src, V=src, mask=mask
            )
            x = x + self.dropout(cross_attn)

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x, self_attn_weights, cross_attn_weights

    def forward_incremental(
        self,
        x: Tensor,
        cache: TransformerLayerCache,
        max_context: int,
        src: Optional[Tensor] = None,
    ) -> Tensor:
        z = self.ln_self_attn(x)
        self_attn, _ = self.mha.forward_incremental(
            Q=z,
            K=z,
            V=z,
            cache=cache.self_attention,
            max_context=max_context,
        )
        x = x + self.dropout(self_attn)

        if self.cross_attention:
            if src is None or cache.cross_attention is None:
                raise ValueError("Cross-attention streaming requires source and cache")
            z = self.ln_src_attn(x)
            src = self.ln_cross_src(src)
            cross_attn, _ = self.mha_cross.forward_incremental(
                Q=z,
                K=src,
                V=src,
                cache=cache.cross_attention,
                max_context=max_context,
            )
            x = x + self.dropout(cross_attn)

        return x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))


class TransformerStereoLayer(TransformerLayer):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # sa1w: self-attention-weights 1
        # ca1w: cross-attention-weights 1
        z1, sa1w, ca1w = super().forward(x=x1, src=x2, mask=mask)
        z2, sa2w, ca2w = super().forward(x=x2, src=x1, mask=mask)
        return z1, z2, [sa1w, ca1w, sa2w, ca2w]


class GPT(nn.Module):
    """
    GPT like transformer Decoder-only class.

    * Uses AliBi attention (no positional embeddings or max-sequence-length)
    """

    def __init__(
        self,
        dim: int = 256,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
        positional_encoding: str = "alibi",
        attention_scale: str = "model",
        normalize_cross_source: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.attention_scale = attention_scale
        self.normalize_cross_source = normalize_cross_source

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    positional_encoding=self.positional_encoding,
                    attention_scale=self.attention_scale,
                    normalize_cross_source=self.normalize_cross_source,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, x: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        all_attention = []

        for layer in self.layers:
            x, self_attn_weights, _ = layer(x)
            if attention:
                all_attention.append(self_attn_weights)

        ret = {"x": x}

        if attention:
            self_attn_weights = torch.stack(all_attention, dim=1)
            ret["attn"] = self_attn_weights

        return ret


class GPTStereo(GPT):
    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerStereoLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=True,
                    positional_encoding=self.positional_encoding,
                    attention_scale=self.attention_scale,
                    normalize_cross_source=self.normalize_cross_source,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        self.combinator = Combinator(dim=self.dim, activation="GELU")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:

        self_attn_a = []
        self_attn_b = []
        cross_attn_a = []
        cross_attn_b = []
        for layer in self.layers:
            x1, x2, attn_list = layer(x1=x1, x2=x2)
            if attention:
                # [sa1w, ca1w, sa2w, ca2w] = attn_list
                self_attn_a.append(attn_list[0])
                cross_attn_a.append(attn_list[1])
                self_attn_b.append(attn_list[2])
                cross_attn_b.append(attn_list[3])

        x = self.combinator(x1, x2)
        ret = {"x": x, "x1": x1, "x2": x2}

        if attention:
            # B, num_layers, num_heads, N, N
            self_attn_a = torch.stack(self_attn_a, dim=1)  # stack on layer dim
            self_attn_b = torch.stack(self_attn_b, dim=1)  # stack on layer dim
            cross_attn_a = torch.stack(cross_attn_a, dim=1)  # stack on layer dim
            cross_attn_b = torch.stack(cross_attn_b, dim=1)  # stack on layer dim
            ret["self_attn"] = torch.stack([self_attn_a, self_attn_b], dim=1)
            ret["cross_attn"] = torch.stack([cross_attn_a, cross_attn_b], dim=1)
        return ret


class Combinator(nn.Module):
    """
    Combines the "ego-centric" representations from identical 'towers'
    processing channel 0 and 1. The towers are identical (shared weights)
    and therefore channel agnostic, e.g. they don't know if they process information
    from the view of speaker A or B.

    Here we have specific layers associated with each channel to join the representations
    into a single coherent space with channel information included.
    """

    def __init__(self, dim: int, activation: str = "GELU"):
        super().__init__()
        self.dim = dim

        # Channel information
        self.h0_a = nn.Linear(dim, dim, bias=False)  # Channel 0
        self.h0_b = nn.Linear(dim, dim, bias=False)  # Channel 1
        self.ln = nn.LayerNorm(self.dim)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Combines the hidden states from identical 'towers' which have processed
        each channel from an 'ego-centric' view. However, the towers are channel agnostic
        by default (shared weights) so in this step we process the information from channel 0, 1
        separately into a joint representation.

        The final representation will (see GPTStereo -> ProjectionModel) go into a final linear
        layer to produce logits.
        """

        # Channel specific information
        ha = self.activation(self.ln(self.h0_a(x1)))
        hb = self.activation(self.ln(self.h0_b(x2)))
        h = ha + hb  # combine estimations from both parties
        return h


class TransformerStereo(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        self_layers: int = 1,
        cross_layers: int = 3,
        num_heads: int = 4,
        dff_k: int = 3,
        dropout: float = 0.1,
        positional_encoding: str = "alibi",
        attention_scale: str = "model",
        normalize_cross_source: bool = False,
    ) -> None:
        super().__init__()
        assert self_layers > 0, f"Must have at least one self layer. got {self_layers}"
        assert (
            cross_layers > 0
        ), f"Must have at least one cross layer. got {cross_layers}"
        self.dim = dim
        self.self_layers = self_layers
        self.cross_layers = cross_layers
        self.num_heads = num_heads
        self.dff_k = dff_k
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.attention_scale = attention_scale
        self.normalize_cross_source = normalize_cross_source

        # Single channel
        self.ar_channel = GPT(
            dim=dim,
            dff_k=dff_k,
            num_layers=self_layers,
            num_heads=num_heads,
            dropout=dropout,
            positional_encoding=positional_encoding,
            attention_scale=attention_scale,
            normalize_cross_source=normalize_cross_source,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=dim,
            dff_k=dff_k,
            num_layers=cross_layers,
            num_heads=num_heads,
            dropout=dropout,
            positional_encoding=positional_encoding,
            attention_scale=attention_scale,
            normalize_cross_source=normalize_cross_source,
        )

    def forward(
        self, x1: Tensor, x2: Tensor, attention: bool = False
    ) -> Mapping[str, Tensor]:
        o1 = self.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.ar_channel(x2, attention=attention)  # ["x"]
        out = self.ar(o1["x"], o2["x"], attention=attention)

        if attention:
            out["cross_self_attn"] = out["self_attn"]
            out["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
            out["cross_attn"] = out["cross_attn"]
        return out

    def init_streaming_cache(
        self, batch_size: int, max_context: int
    ) -> TransformerStereoCache:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_context <= 0:
            raise ValueError(f"max_context must be positive, got {max_context}")
        return TransformerStereoCache(
            batch_size=batch_size,
            max_context=max_context,
            channel_layers=[
                TransformerLayerCache() for _ in self.ar_channel.layers
            ],
            stereo_layers=[
                TransformerLayerCache(cross_attention=KVCache())
                for _ in self.ar.layers
            ],
        )

    def forward_incremental(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: TransformerStereoCache,
    ) -> Mapping[str, Tensor]:
        """Process only new frames while updating bounded stereo KV caches."""
        if self.training:
            raise RuntimeError("Incremental TransformerStereo inference requires eval()")
        if x1.shape != x2.shape or x1.ndim != 3:
            raise ValueError(
                f"Expected aligned [batch, time, dim] inputs, got {x1.shape}, {x2.shape}"
            )
        if x1.shape[0] != cache.batch_size:
            raise ValueError(
                f"Streaming cache batch size {cache.batch_size} != {x1.shape[0]}"
            )

        batch_size = cache.batch_size
        channels = torch.cat((x1, x2), dim=0)
        for layer, layer_cache in zip(
            self.ar_channel.layers, cache.channel_layers
        ):
            channels = layer.forward_incremental(
                channels, layer_cache, cache.max_context
            )
        x1, x2 = channels.split(batch_size, dim=0)

        for layer, layer_cache in zip(self.ar.layers, cache.stereo_layers):
            channels = torch.cat((x1, x2), dim=0)
            sources = torch.cat((x2, x1), dim=0)
            channels = layer.forward_incremental(
                channels,
                layer_cache,
                cache.max_context,
                src=sources,
            )
            x1, x2 = channels.split(batch_size, dim=0)

        return {"x": self.ar.combinator(x1, x2), "x1": x1, "x2": x2}


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        self_layers: int = 1,
        cross_layers: int = 3,
        num_heads: int = 4,
        dff_k: int = 3,
        dropout: float = 0.1,
        positional_encoding: str = "alibi",
        attention_scale: str = "model",
        normalize_cross_source: bool = False,
    ) -> None:
        super().__init__()
        assert self_layers > 0, f"Must have at least one self layer. got {self_layers}"
        assert (
            cross_layers > 0
        ), f"Must have at least one cross layer. got {cross_layers}"
        self.dim = dim
        self.self_layers = self_layers
        self.cross_layers = cross_layers
        self.num_heads = num_heads
        self.dff_k = dff_k
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        self.attention_scale = attention_scale
        self.normalize_cross_source = normalize_cross_source

        # Single channel
        self.ar_channel = GPT(
            dim=dim,
            dff_k=dff_k,
            num_layers=self_layers,
            num_heads=num_heads,
            dropout=dropout,
            positional_encoding=positional_encoding,
            attention_scale=attention_scale,
            normalize_cross_source=normalize_cross_source,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=dim,
            dff_k=dff_k,
            num_layers=cross_layers,
            num_heads=num_heads,
            dropout=dropout,
            positional_encoding=positional_encoding,
            attention_scale=attention_scale,
            normalize_cross_source=normalize_cross_source,
        )

    def forward(
        self, x1: Tensor, x2: Tensor, attention: bool = False
    ) -> Mapping[str, Tensor]:
        o1 = self.ar_channel(x1, attention=attention)  # ["x"]
        o2 = self.ar_channel(x2, attention=attention)  # ["x"]
        out = self.ar(o1["x"], o2["x"], attention=attention)

        if attention:
            out["cross_self_attn"] = out["self_attn"]
            out["self_attn"] = torch.stack([o1["attn"], o2["attn"]], dim=1)
            out["cross_attn"] = out["cross_attn"]
        return out


class ProjectionLayer(nn.Module):
    """
    Inpired by huggingface transformers `Wav2Vec2FeatureProjection`
    in `transformers/models/wav2vec2/modeling_wav2vec2.py`
    """

    def __init__(self, inp_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(inp_dim)
        self.projection = nn.Linear(inp_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.dropout(x)


class VACondition(nn.Module):
    def __init__(
        self, dim: int, va_history: bool = False, va_history_bins: int = 5
    ) -> None:
        super().__init__()
        self.dim = dim
        self.va_history = va_history
        self.va_history_bins = va_history_bins
        self.va_cond = nn.Linear(2, dim)  # va: 2 one-hot encodings -> dim
        self.ln = nn.LayerNorm(dim)
        if va_history:
            # vah: (N, vah_bins) -> dim
            self.va_hist_cond = nn.Linear(va_history_bins, dim)

    def forward(
        self, vad: torch.Tensor, va_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        v_cond = self.va_cond(vad)

        # Add vad-history information
        if self.va_history and va_history is not None:
            v_cond += self.va_hist_cond(va_history)

        return self.ln(v_cond)


def test_gpt():
    model = GPT(dim=256, dff_k=3, num_layers=4, num_heads=8)
    x = torch.rand((4, 20, model.dim))
    with torch.no_grad():
        z, attn = model(x, attention=True)
    print("z: ", tuple(z.shape))
    print("attn: ", tuple(attn.shape))
    b = 0
    fig, ax = plt.subplots(
        model.num_heads, model.num_layers, sharex=True, sharey=True, figsize=(12, 12)
    )
    for n_layer in range(model.num_layers):
        for n_head in range(model.num_heads):
            ax[n_head, n_layer].imshow(
                attn[b, n_layer, n_head],
                aspect="auto",
                origin="upper",
                interpolation="none",
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            if n_layer == 0:
                ax[n_head, n_layer].set_ylabel(f"Head {n_head}")
            if n_head == 0:
                ax[n_head, n_layer].set_title(f"Layer {n_layer}")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    pass
