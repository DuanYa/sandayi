from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Protocol

from sandayi.ai.features import ACTION_SIZE, BID_ACTIONS, CARD_RANK_VALUE, CARD_VOCAB, CARD_TO_ID, SCALAR_SIZE, SUIT_ACTIONS, SUIT_ORDER, action_to_bid, action_to_suit, bid_to_action, card_to_action, encode_state, suit_to_action
from sandayi.ai.network import create_policy_model

logger = logging.getLogger(__name__)


class AIStrategy(Protocol):
    def decide(self, state: dict[str, Any]) -> dict[str, Any] | None:
        ...


class RuleBasedStrategy:
    def decide(self, state: dict[str, Any]) -> dict[str, Any] | None:
        legal = state.get("legal_actions", {})
        action_type = legal.get("type")
        hand = state.get("hand", [])
        if action_type == "bid":
            return self._decide_bid(state, hand, legal.get("bids", []))
        if action_type == "bury":
            cards = self._choose_bury_cards(hand, legal.get("count", 6))
            return {"type": "bury", "cards": cards}
        if action_type == "trump":
            return {"type": "trump", "suit": self._choose_trump(hand)}
        if action_type == "play":
            cards = legal.get("cards", [])
            if cards:
                return {"type": "play", "card": self._choose_play_card(state, cards)}
        return None

    def _decide_bid(self, state: dict[str, Any], hand: list[str], bids: list[int]) -> dict[str, Any]:
        if not bids:
            return {"type": "bid", "bid": None}
        strength = self._hand_strength(hand)
        max_bid = 0
        if strength >= 0.58:
            max_bid = 80
        elif strength >= 0.50:
            max_bid = 75
        elif strength >= 0.42:
            max_bid = 70
        elif strength >= 0.34:
            max_bid = 65
        elif strength >= 0.26:
            max_bid = 60
        affordable = [bid for bid in bids if bid <= max_bid]
        if not affordable:
            return {"type": "bid", "bid": None}
        return {"type": "bid", "bid": max(affordable)}

    def _choose_bury_cards(self, hand: list[str], count: int) -> list[str]:
        ordered = sorted(hand, key=self._bury_value)
        return ordered[:count]

    def _choose_trump(self, hand: list[str]) -> str:
        counts = {s: 0.0 for s in SUIT_ORDER}
        for card in hand:
            if card and card[0] in counts:
                counts[card[0]] += 1.0 + max(0, self._card_value(card) - 10) * 0.15
        return max(SUIT_ORDER, key=lambda s: counts[s])

    def _choose_play_card(self, state: dict[str, Any], cards: list[str]) -> str:
        banker_id = state.get("banker_id")
        players = state.get("players", [])
        hand_count = next((p.get("hand_count", 0) for p in players if p.get("id") == state.get("turn_player_id")), len(state.get("hand", [])))
        is_last_card = hand_count <= 1
        legal_sorted = sorted(cards, key=self._card_value)
        has_points = [card for card in legal_sorted if self._point_value(card) > 0]
        is_banker_turn = state.get("turn_player_id") == banker_id
        farmers_points = int(state.get("farmers_points", 0))
        bid = int(state.get("highest_bid") or 60)
        farmers_need = 100 - bid
        if is_last_card and not is_banker_turn:
            return max(legal_sorted, key=self._card_value)
        if is_banker_turn:
            if farmers_points >= farmers_need - 10:
                return max(legal_sorted, key=self._card_value)
            return min(legal_sorted, key=lambda c: (self._point_value(c), self._card_value(c)))
        if has_points and farmers_points < farmers_need:
            return min(has_points, key=self._card_value)
        return min(legal_sorted, key=self._card_value)

    def _hand_strength(self, hand: list[str]) -> float:
        if not hand:
            return 0.0
        score_cards = sum(self._point_value(card) for card in hand) / 100.0
        high_cards = sum(1 for card in hand if self._card_value(card) >= 13) / max(1, len(hand))
        trump_like = sum(1 for card in hand if card in ("XJ", "BJ") or card.endswith("2")) / max(1, len(hand))
        suit_balance = max(sum(1 for card in hand if card.startswith(suit)) for suit in SUIT_ORDER) / max(1, len(hand))
        return min(1.0, score_cards * 0.35 + high_cards * 0.25 + trump_like * 0.25 + suit_balance * 0.15)

    def _bury_value(self, card: str) -> tuple[int, int]:
        return (self._point_value(card), self._card_value(card))

    def _point_value(self, card: str) -> int:
        if card.endswith("5"):
            return 5
        if card.endswith("10") or card.endswith("K"):
            return 10
        return 0

    def _card_value(self, card: str) -> int:
        if card in ("XJ", "BJ"):
            return CARD_RANK_VALUE[card]
        return CARD_RANK_VALUE.get(card[1:], 0)

class TransformerRLStrategy:
    def __init__(self, model_path: str | None = None, fallback: AIStrategy | None = None) -> None:
        self.fallback = fallback or RuleBasedStrategy()
        self.model_path = Path(model_path or os.getenv("SANDAYI_AI_MODEL", "models/sandayi_transformer_policy.pt"))
        self.torch = None
        self.model = None
        self.device = "cpu"
        self._load_model()

    def decide(self, state: dict[str, Any]) -> dict[str, Any] | None:
        legal = state.get("legal_actions", {})
        action_type = legal.get("type")
        if self.model is None or self.torch is None:
            return self.fallback.decide(state)
        try:
            if action_type == "play" and legal.get("cards"):
                card = self._predict_card(state, legal["cards"])
                if card:
                    return {"type": "play", "card": card}
            if action_type == "bid":
                bid = self._predict_bid(state, legal.get("bids", []))
                if bid is None or bid in legal.get("bids", []):
                    return {"type": "bid", "bid": bid}
            if action_type == "trump":
                suit = self._predict_trump(state, legal.get("suits", []))
                if suit:
                    return {"type": "trump", "suit": suit}
            if action_type == "bury":
                cards = self._predict_bury_cards(state, legal.get("count", 6))
                if cards:
                    return {"type": "bury", "cards": cards}
        except Exception:
            logger.exception("Transformer RL 推理失败，回退到规则策略")
        return self.fallback.decide(state)

    def _load_model(self) -> None:
        if importlib.util.find_spec("torch") is None:
            logger.info("未安装 torch，AI 使用规则策略兜底")
            return
        import torch
        from torch import nn

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = create_policy_model(nn, vocab_size=len(CARD_VOCAB) + 1, action_size=ACTION_SIZE, scalar_size=SCALAR_SIZE).to(self.device)
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model_state = self.model.state_dict()
            compatible_state = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}
            missing_or_changed = sorted(set(model_state) - set(compatible_state))
            self.model.load_state_dict(compatible_state, strict=False)
            if missing_or_changed:
                logger.warning("模型结构已变更，部分参数使用初始化值 path=%s missing_or_changed=%s", self.model_path, missing_or_changed[:8])
            logger.info("已加载 Transformer RL AI 模型 path=%s device=%s", self.model_path, self.device)
        else:
            logger.warning("未找到 Transformer RL AI 模型 path=%s，使用规则策略兜底", self.model_path)
        self.model.eval()

    def _predict_bid(self, state: dict[str, Any], legal_bids: list[int]) -> int | None:
        assert self.torch is not None and self.model is not None
        encoded = encode_state(state)
        cards = self.torch.tensor([encoded.card_tokens], dtype=self.torch.long, device=self.device)
        scalars = self.torch.tensor([encoded.scalar_features], dtype=self.torch.float32, device=self.device)
        segments = self.torch.tensor([encoded.segment_tokens], dtype=self.torch.long, device=self.device)
        mask = self.torch.tensor(encoded.action_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _ = self.model(cards, scalars, segments)
            logits = logits[0].masked_fill(~mask, -1e4)
        candidate_actions = [bid_to_action(None)] + [bid_to_action(bid) for bid in legal_bids if bid in BID_ACTIONS]
        candidate_actions = [idx for idx in candidate_actions if bool(mask[idx].item())]
        if not candidate_actions:
            return None
        best_idx = max(candidate_actions, key=lambda idx: float(logits[idx].item()))
        return action_to_bid(best_idx)

    def _masked_logits(self, state: dict[str, Any]) -> tuple[Any, Any]:
        assert self.torch is not None and self.model is not None
        encoded = encode_state(state)
        cards = self.torch.tensor([encoded.card_tokens], dtype=self.torch.long, device=self.device)
        scalars = self.torch.tensor([encoded.scalar_features], dtype=self.torch.float32, device=self.device)
        segments = self.torch.tensor([encoded.segment_tokens], dtype=self.torch.long, device=self.device)
        mask = self.torch.tensor(encoded.action_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _ = self.model(cards, scalars, segments)
            logits = logits[0].masked_fill(~mask, -1e4)
        return logits, mask

    def _predict_trump(self, state: dict[str, Any], legal_suits: list[str]) -> str | None:
        logits, mask = self._masked_logits(state)
        candidate_actions = [suit_to_action(suit) for suit in legal_suits if suit in SUIT_ACTIONS]
        candidate_actions = [idx for idx in candidate_actions if bool(mask[idx].item())]
        if not candidate_actions:
            return None
        best_idx = max(candidate_actions, key=lambda idx: float(logits[idx].item()))
        return action_to_suit(best_idx)

    def _predict_bury_cards(self, state: dict[str, Any], count: int) -> list[str]:
        hand = list(state.get("hand", []))
        selected: list[str] = []
        for _ in range(min(count, len(hand))):
            logits, _ = self._masked_logits(state)
            legal_ids = [card_to_action(card) for card in hand if card in CARD_TO_ID]
            if not legal_ids:
                break
            best_idx = max(legal_ids, key=lambda idx: float(logits[idx].item()))
            card = CARD_VOCAB[best_idx]
            selected.append(card)
            hand.remove(card)
        return selected

    def _predict_card(self, state: dict[str, Any], legal_cards: list[str]) -> str | None:
        assert self.torch is not None and self.model is not None
        encoded = encode_state(state)
        cards = self.torch.tensor([encoded.card_tokens], dtype=self.torch.long, device=self.device)
        scalars = self.torch.tensor([encoded.scalar_features], dtype=self.torch.float32, device=self.device)
        segments = self.torch.tensor([encoded.segment_tokens], dtype=self.torch.long, device=self.device)
        mask = self.torch.tensor(encoded.action_mask, dtype=self.torch.bool, device=self.device)
        with self.torch.no_grad():
            logits, _ = self.model(cards, scalars, segments)
            logits = logits[0].masked_fill(~mask, -1e4)
        legal_ids = [CARD_TO_ID[c] - 1 for c in legal_cards if c in CARD_TO_ID]
        if not legal_ids:
            return None
        best_idx = max(legal_ids, key=lambda idx: float(logits[idx].item()))
        return CARD_VOCAB[best_idx]


def create_ai_strategy(model_name: str | None = None) -> AIStrategy:
    if not model_name or model_name == "transformer":
        return TransformerRLStrategy()
    if model_name == "rule":
        return RuleBasedStrategy()
    return TransformerRLStrategy(model_name)
