from __future__ import annotations

from typing import Any

from sandayi.ai.features import MAX_CARD_TOKENS, SEGMENT_SIZE


def create_policy_model(nn: Any, vocab_size: int, action_size: int, scalar_size: int, d_model: int = 512, nhead: int = 8, layers: int = 8) -> Any:
    class SandayiTransformerPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.card_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.position_embedding = nn.Embedding(MAX_CARD_TOKENS, d_model)
            self.segment_embedding = nn.Embedding(SEGMENT_SIZE, d_model, padding_idx=0)
            self.input_norm = nn.LayerNorm(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.attention_pool = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            self.scalar = nn.Sequential(
                nn.LayerNorm(scalar_size),
                nn.Linear(scalar_size, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            fused_size = d_model * 2
            self.fusion = nn.Sequential(
                nn.LayerNorm(fused_size),
                nn.Linear(fused_size, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
            )
            self.policy_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(d_model, action_size),
            )
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(d_model, 1),
            )

        def forward(self, cards: Any, scalars: Any, segments: Any | None = None) -> tuple[Any, Any]:
            import torch

            mask = cards.eq(0)
            if segments is None:
                segments = torch.zeros_like(cards)
            positions = torch.arange(cards.size(1), device=cards.device).unsqueeze(0).expand_as(cards)
            embedded = self.card_embedding(cards) + self.position_embedding(positions) + self.segment_embedding(segments)
            encoded = self.encoder(self.input_norm(embedded), src_key_padding_mask=mask)
            attention_logits = self.attention_pool(encoded).squeeze(-1).masked_fill(mask, -1e4)
            attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
            pooled = (encoded * attention_weights).sum(dim=1)
            scalar_features = self.scalar(scalars)
            features = self.fusion(torch.cat([pooled, scalar_features], dim=-1))
            return self.policy_head(features), self.value_head(features).squeeze(-1)

        def parameter_count(self) -> int:
            return sum(param.numel() for param in self.parameters())

    return SandayiTransformerPolicy()
