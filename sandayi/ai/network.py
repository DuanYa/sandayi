from __future__ import annotations

from typing import Any


def create_policy_model(nn: Any, vocab_size: int, action_size: int, scalar_size: int, d_model: int = 192, nhead: int = 6, layers: int = 4) -> Any:
    class SandayiTransformerPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.card_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.scalar = nn.Sequential(nn.Linear(scalar_size, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.policy_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, action_size))
            self.value_head = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1))

        def forward(self, cards: Any, scalars: Any) -> tuple[Any, Any]:
            import torch

            mask = cards.eq(0)
            encoded = self.encoder(self.card_embedding(cards), src_key_padding_mask=mask)
            valid = (~mask).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            scalar_features = self.scalar(scalars)
            features = torch.cat([pooled, scalar_features], dim=-1)
            return self.policy_head(features), self.value_head(features).squeeze(-1)

        def parameter_count(self) -> int:
            return sum(param.numel() for param in self.parameters())

    return SandayiTransformerPolicy()
