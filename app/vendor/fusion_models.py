"""Study-level fusion heads over the ten swallows of an examination.

All four heads consume exactly the same frozen swallow embeddings, the same
frozen swallow-level logits and the same Chicago v4.0 proxy metrics, so the
comparison isolates the aggregation rule.

``simple``           masked mean of the swallow embeddings.
``transformer``      self-attention over the swallow embeddings.
``chicago``          hierarchical aggregation of swallow-level predictions into
                     Chicago Classification v4.0 study-level features.
``chicago_set``      the Chicago branch plus a complementary Set Transformer
                     branch (SAB encoder + pooling by multihead attention).
"""

from __future__ import annotations

import torch
from torch import nn

FUSIONS = ("simple", "transformer", "multiview", "chicago", "chicago_set", "chicago_multiview")


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)


# ------------------------------------------------------------ Set Transformer


class MultiheadAttentionBlock(nn.Module):
    """MAB(Q, K) from Lee et al., Set Transformer (2019)."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(dim)
        self.norm_feedforward = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        attended, _ = self.attention(query, key, key, key_padding_mask=key_padding_mask)
        hidden = self.norm_attention(query + attended)
        return self.norm_feedforward(hidden + self.feedforward(hidden))


class SetTransformer(nn.Module):
    """SAB encoder followed by pooling by multihead attention (PMA)."""

    def __init__(self, input_dim: int, dim: int = 256, heads: int = 4, blocks: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        self.blocks = nn.ModuleList(
            MultiheadAttentionBlock(dim, heads, dropout) for _ in range(blocks)
        )
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pooling = MultiheadAttentionBlock(dim, heads, dropout)
        self.output_dim = dim

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding = ~mask
        tokens = self.project(embeddings)
        for block in self.blocks:
            tokens = block(tokens, tokens, padding)
        # Zero out padded slots so the pooling query cannot read them.
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        seeds = self.seed.expand(tokens.shape[0], -1, -1)
        return self.pooling(seeds, tokens, padding).squeeze(1)


# -------------------------------------------- Transformer multi-view network


class MultiViewTransformer(nn.Module):
    """Transformer-based multi-view network over the ten swallows of a study.

    The plain self-attention baseline in ``FusionModel('transformer')`` has two
    design faults that cost it accuracy: it adds a sinusoidal-style positional
    embedding over slots that are largely exchangeable, and it collapses the
    contextualised tokens with an unweighted masked mean, which dilutes the one
    or two abnormal swallows that carry the diagnosis.

    A multi-view network instead treats each swallow as a *view* of the same
    examination: views are tagged with a learned view-identity embedding rather
    than a positional code, cross-view attention lets them exchange evidence,
    and a dedicated classification token aggregates them by attention, so a
    single premature or hypercontractile swallow can dominate the study
    representation.  The class token is concatenated with the masked mean, which
    keeps a stable global summary next to the selective one.
    """

    def __init__(
        self, input_dim: int, dim: int = 256, heads: int = 4, blocks: int = 3, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        # View identity, not sequence position: initialised near zero so the
        # network starts permutation-invariant and only uses swallow order if
        # that turns out to help.
        self.view_embedding = nn.Parameter(torch.zeros(1, 10, dim))
        nn.init.trunc_normal_(self.view_embedding, std=0.01)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=blocks, norm=nn.LayerNorm(dim))
        self.output_dim = dim * 2

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, views = embeddings.shape[:2]
        tokens = self.project(embeddings) + self.view_embedding[:, :views]
        tokens = torch.cat([self.class_token.expand(batch, -1, -1), tokens], dim=1)
        # The class token is never padded, so no row is fully masked.
        padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1
        )
        contextual = self.encoder(tokens, src_key_padding_mask=padding)
        pooled = masked_mean(contextual[:, 1:], mask)
        return torch.cat([contextual[:, 0], pooled], dim=-1)


# ------------------------------------------------------------- Chicago branch


class ChicagoAggregator(nn.Module):
    """Chicago Classification v4.0 style hierarchical aggregation.

    The frozen swallow classifier supplies per-swallow diagnostic posteriors.
    Those are collapsed into study-level statistics that mirror how a reader
    applies CC v4.0: what fraction of the ten swallows shows each abnormal
    pattern, how confident the strongest swallow is, and where the summary
    pressure metrics (median IRP, DCI distribution, distal latency) fall with
    respect to the published cut-offs.
    """

    def __init__(self, num_classes: int, metric_dim: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        # per class: mean posterior, max posterior, min posterior, argmax share,
        # share above 0.5 -- five study-level descriptors per diagnosis.
        self.prediction_dim = num_classes * 5
        self.metric_dim = metric_dim
        self.encoder = _mlp(self.prediction_dim + metric_dim, hidden, hidden, dropout)
        self.output_dim = hidden

    @staticmethod
    def summarise(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        weights = mask.to(probabilities.dtype).unsqueeze(-1)
        count = weights.sum(1).clamp_min(1.0)

        mean = (probabilities * weights).sum(1) / count
        filled_low = probabilities.masked_fill(~mask.unsqueeze(-1), 1.0)
        filled_high = probabilities.masked_fill(~mask.unsqueeze(-1), 0.0)
        maximum = filled_high.max(dim=1).values
        minimum = filled_low.min(dim=1).values

        winners = torch.zeros_like(probabilities).scatter_(
            -1, probabilities.argmax(-1, keepdim=True), 1.0
        )
        argmax_share = (winners * weights).sum(1) / count
        confident_share = ((probabilities > 0.5).to(probabilities.dtype) * weights).sum(1) / count
        return torch.cat([mean, maximum, minimum, argmax_share, confident_share], dim=-1)

    def forward(
        self, swallow_logits: torch.Tensor, mask: torch.Tensor, study_metrics: torch.Tensor
    ) -> torch.Tensor:
        summary = self.summarise(swallow_logits, mask)
        return self.encoder(torch.cat([summary, study_metrics], dim=-1))


# --------------------------------------------------------------- fusion heads


class FusionModel(nn.Module):
    def __init__(
        self,
        fusion: str,
        embedding_dim: int,
        num_classes: int,
        metric_dim: int,
        *,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        if fusion == "simple":
            self.norm = nn.LayerNorm(embedding_dim)
            self.classifier = _mlp(embedding_dim, hidden, num_classes, dropout)
        elif fusion == "transformer":
            self.project = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.LayerNorm(hidden))
            self.position = nn.Parameter(torch.zeros(1, 10, hidden))
            nn.init.trunc_normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(hidden))
            self.classifier = _mlp(hidden, hidden, num_classes, dropout)
        elif fusion == "multiview":
            self.multiview = MultiViewTransformer(embedding_dim, hidden, 4, 3, dropout)
            self.classifier = _mlp(self.multiview.output_dim, hidden, num_classes, dropout)
        elif fusion == "chicago":
            self.chicago = ChicagoAggregator(num_classes, metric_dim, hidden, dropout)
            self.classifier = _mlp(self.chicago.output_dim, hidden, num_classes, dropout)
        elif fusion == "chicago_multiview":
            self.chicago = ChicagoAggregator(num_classes, metric_dim, hidden, dropout)
            self.multiview = MultiViewTransformer(embedding_dim, hidden, 4, 3, dropout)
            self.classifier = _mlp(
                self.chicago.output_dim + self.multiview.output_dim, hidden, num_classes, dropout
            )
        elif fusion == "chicago_set":
            self.chicago = ChicagoAggregator(num_classes, metric_dim, hidden, dropout)
            self.set_transformer = SetTransformer(embedding_dim, hidden, 4, 2, dropout)
            self.classifier = _mlp(
                self.chicago.output_dim + self.set_transformer.output_dim, hidden, num_classes, dropout
            )
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(
        self,
        embeddings: torch.Tensor,
        swallow_logits: torch.Tensor,
        mask: torch.Tensor,
        study_metrics: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion == "simple":
            return self.classifier(self.norm(masked_mean(embeddings, mask)))
        if self.fusion == "transformer":
            tokens = self.project(embeddings) + self.position[:, : embeddings.shape[1]]
            # Fully padded rows never occur: every exam keeps at least one swallow.
            contextual = self.encoder(tokens, src_key_padding_mask=~mask)
            return self.classifier(masked_mean(contextual, mask))
        if self.fusion == "multiview":
            return self.classifier(self.multiview(embeddings, mask))
        if self.fusion == "chicago":
            return self.classifier(self.chicago(swallow_logits, mask, study_metrics))
        chicago = self.chicago(swallow_logits, mask, study_metrics)
        branch = (
            self.multiview(embeddings, mask)
            if self.fusion == "chicago_multiview"
            else self.set_transformer(embeddings, mask)
        )
        return self.classifier(torch.cat([chicago, branch], dim=-1))
