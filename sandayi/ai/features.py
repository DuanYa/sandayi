from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUIT_ORDER = ["S", "H", "C", "D"]
RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
CARD_VOCAB = [f"{s}{r}" for s in SUIT_ORDER for r in RANKS] + ["XJ", "BJ"]
CARD_TO_ID = {card: idx + 1 for idx, card in enumerate(CARD_VOCAB)}
CARD_RANK_VALUE = {"3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14, "2": 15, "XJ": 16, "BJ": 17}
ID_TO_CARD = {idx: card for card, idx in CARD_TO_ID.items()}
BID_ACTIONS = [None, 60, 65, 70, 75, 80]
BID_ACTION_OFFSET = len(CARD_VOCAB)
ACTION_SIZE = len(CARD_VOCAB) + len(BID_ACTIONS)
MAX_CARD_TOKENS = 128
SCALAR_SIZE = 29


@dataclass
class EncodedState:
    card_tokens: list[int]
    scalar_features: list[float]
    action_mask: list[float]


def encode_state(state: dict[str, Any]) -> EncodedState:
    hand = state.get("hand", [])
    current_plays = [p.get("card", "") for p in (state.get("current_trick") or {}).get("plays", [])]
    last_plays = [p.get("card", "") for p in (state.get("last_trick") or {}).get("plays", [])]
    history_plays = [p.get("card", "") for trick in state.get("completed_trick_plays", []) for p in trick]
    kitty = state.get("kitty", [])
    tokens = [CARD_TO_ID.get(card, 0) for card in hand + current_plays + last_plays + history_plays + kitty]
    tokens = (tokens + [0] * MAX_CARD_TOKENS)[:MAX_CARD_TOKENS]
    legal = state.get("legal_actions", {})
    legal_cards = legal.get("cards", []) if legal.get("type") == "play" else []
    mask = [0.0] * ACTION_SIZE
    if legal.get("type") == "play":
        for card in legal_cards:
            card_id = CARD_TO_ID.get(card)
            if card_id:
                mask[card_id - 1] = 1.0
    elif legal.get("type") == "bid":
        mask[bid_to_action(None)] = 1.0
        for bid in legal.get("bids", []):
            mask[bid_to_action(bid)] = 1.0
    players = state.get("players", [])
    viewer_id = state.get("viewer_id")
    banker_id = state.get("banker_id")
    turn_id = state.get("turn_player_id")
    hand_counts = [float(p.get("hand_count", 0)) / 18.0 for p in players[:4]]
    hand_counts = (hand_counts + [0.0] * 4)[:4]
    bid_values = [float((record.get("bid") or 0)) / 80.0 for record in state.get("bid_records", [])[-4:]]
    bid_values = ([0.0] * (4 - len(bid_values)) + bid_values)[-4:]
    trump = state.get("trump")
    trump_one_hot = [1.0 if trump == suit else 0.0 for suit in SUIT_ORDER]
    phase = state.get("phase")
    scalars = [
        float(state.get("banker_points", 0)) / 120.0,
        float(state.get("farmers_points", 0)) / 120.0,
        float(state.get("highest_bid") or 0) / 80.0,
        float(len(hand)) / 18.0,
        float(len(legal_cards)) / 18.0,
        1.0 if banker_id else 0.0,
        1.0 if viewer_id and viewer_id == banker_id else 0.0,
        1.0 if viewer_id and viewer_id != banker_id and banker_id else 0.0,
        1.0 if viewer_id and viewer_id == turn_id else 0.0,
        float(state.get("completed_tricks", 0)) / 12.0,
        float(len(current_plays)) / 4.0,
        1.0 if phase == "bidding" else 0.0,
        1.0 if phase == "trump" else 0.0,
        1.0 if phase == "kitty" else 0.0,
        1.0 if phase == "playing" else 0.0,
        1.0 if state.get("kitty_view_allowed") else 0.0,
        float(len(kitty)) / 6.0,
    ] + trump_one_hot + hand_counts + bid_values
    return EncodedState(tokens, scalars, mask)


def card_to_action(card: str) -> int:
    return CARD_TO_ID[card] - 1


def action_to_card(action: int) -> str:
    return CARD_VOCAB[action]


def bid_to_action(bid: int | None) -> int:
    return BID_ACTION_OFFSET + BID_ACTIONS.index(bid)


def action_to_bid(action: int) -> int | None:
    return BID_ACTIONS[action - BID_ACTION_OFFSET]


def is_card_action(action: int) -> bool:
    return 0 <= action < len(CARD_VOCAB)


def is_bid_action(action: int) -> bool:
    return BID_ACTION_OFFSET <= action < ACTION_SIZE
