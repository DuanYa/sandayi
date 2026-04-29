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
SUIT_ACTIONS = SUIT_ORDER
SUIT_ACTION_OFFSET = BID_ACTION_OFFSET + len(BID_ACTIONS)
ACTION_SIZE = SUIT_ACTION_OFFSET + len(SUIT_ACTIONS)
MAX_CARD_TOKENS = 128
SCALAR_SIZE = 33
# segment encoding: area × player_relation
# 0=pad 1=hand 2~4=current_trick(self/banker/ally)
# 5~7=last_trick(self/banker/ally) 8~10=history(self/banker/ally) 11=kitty
SEGMENT_SIZE = 12
SEGMENT_PAD = 0
SEGMENT_HAND = 1
_SEG_CT_SELF = 2
_SEG_CT_BANKER = 3
_SEG_CT_ALLY = 4
_SEG_LT_SELF = 5
_SEG_LT_BANKER = 6
_SEG_LT_ALLY = 7
_SEG_HI_SELF = 8
_SEG_HI_BANKER = 9
_SEG_HI_ALLY = 10
SEGMENT_KITTY = 11


def _player_segment(player_id: str, viewer_id: str | None, banker_id: str | None, base_self: int, base_banker: int, base_ally: int) -> int:
    if viewer_id and player_id == viewer_id:
        return base_self
    if banker_id and player_id == banker_id:
        return base_banker
    return base_ally


@dataclass
class EncodedState:
    card_tokens: list[int]
    scalar_features: list[float]
    action_mask: list[float]
    segment_tokens: list[int]


def encode_state(state: dict[str, Any]) -> EncodedState:
    hand = state.get("hand", [])
    viewer_id = state.get("viewer_id")
    banker_id = state.get("banker_id")
    current_trick_data = state.get("current_trick") or {}
    current_trick_plays = current_trick_data.get("plays", [])
    last_trick_data = state.get("last_trick") or {}
    last_trick_plays = last_trick_data.get("plays", [])
    history_tricks = state.get("completed_trick_plays", [])
    kitty = state.get("kitty", [])
    cards_with_segments: list[tuple[str, int]] = []
    for card in hand:
        cards_with_segments.append((card, SEGMENT_HAND))
    for p in current_trick_plays:
        seg = _player_segment(p.get("player_id", ""), viewer_id, banker_id, _SEG_CT_SELF, _SEG_CT_BANKER, _SEG_CT_ALLY)
        cards_with_segments.append((p.get("card", ""), seg))
    for p in last_trick_plays:
        seg = _player_segment(p.get("player_id", ""), viewer_id, banker_id, _SEG_LT_SELF, _SEG_LT_BANKER, _SEG_LT_ALLY)
        cards_with_segments.append((p.get("card", ""), seg))
    for trick in history_tricks:
        for p in trick:
            seg = _player_segment(p.get("player_id", ""), viewer_id, banker_id, _SEG_HI_SELF, _SEG_HI_BANKER, _SEG_HI_ALLY)
            cards_with_segments.append((p.get("card", ""), seg))
    for card in kitty:
        cards_with_segments.append((card, SEGMENT_KITTY))
    cards_with_segments = cards_with_segments[:MAX_CARD_TOKENS]
    tokens = [CARD_TO_ID.get(card, 0) for card, _ in cards_with_segments]
    segments = [seg for _, seg in cards_with_segments]
    tokens = (tokens + [0] * MAX_CARD_TOKENS)[:MAX_CARD_TOKENS]
    segments = (segments + [SEGMENT_PAD] * MAX_CARD_TOKENS)[:MAX_CARD_TOKENS]
    legal = state.get("legal_actions", {})
    legal_cards = legal.get("cards", []) if legal.get("type") == "play" else []
    mask = [0.0] * ACTION_SIZE
    if legal.get("type") == "play":
        for card in legal_cards:
            card_id = CARD_TO_ID.get(card)
            if card_id:
                mask[card_id - 1] = 1.0
    elif legal.get("type") == "bury":
        for card in hand:
            card_id = CARD_TO_ID.get(card)
            if card_id:
                mask[card_id - 1] = 1.0
    elif legal.get("type") == "bid":
        mask[bid_to_action(None)] = 1.0
        for bid in legal.get("bids", []):
            mask[bid_to_action(bid)] = 1.0
    elif legal.get("type") == "trump":
        for suit in legal.get("suits", SUIT_ORDER):
            mask[suit_to_action(suit)] = 1.0
    players = state.get("players", [])
    turn_id = state.get("turn_player_id")
    player_ids = [p.get("id", "") for p in players[:4]]
    hand_counts = [float(p.get("hand_count", 0)) / 18.0 for p in players[:4]]
    hand_counts = (hand_counts + [0.0] * 4)[:4]
    bid_values = [float((record.get("bid") or 0)) / 80.0 for record in state.get("bid_records", [])[-4:]]
    bid_values = ([0.0] * (4 - len(bid_values)) + bid_values)[-4:]
    trump = state.get("trump")
    trump_one_hot = [1.0 if trump == suit else 0.0 for suit in SUIT_ORDER]
    phase = state.get("phase")
    viewer_idx = player_ids.index(viewer_id) if viewer_id and viewer_id in player_ids else 0
    banker_idx = player_ids.index(banker_id) if banker_id and banker_id in player_ids else 0
    relative_seat = (viewer_idx - banker_idx) % 4 if banker_id else 0
    seat_one_hot = [1.0 if relative_seat == i else 0.0 for i in range(4)]
    current_play_order = float(len(current_trick_plays)) / 4.0
    # scalars: 33 dimensions
    # [0] banker_points/120  [1] farmers_points/120  [2] highest_bid/80
    # [3] hand_size/18  [4] legal_card_count/18
    # [5] has_banker  [6] is_banker  [7] is_farmer  [8] is_my_turn
    # [9] completed_tricks/12  [10] current_trick_play_order/4
    # [11~14] phase one-hot (bidding/trump/kitty/playing)
    # [15] kitty_visible  [16] kitty_count/6
    # [17~20] trump one-hot (S/H/C/D)
    # [21~24] hand_counts per player /18
    # [25~28] bid_records /80
    # [29~32] relative_seat one-hot (0=banker, 1=banker+1, 2=banker+2, 3=banker+3)
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
        current_play_order,
        1.0 if phase == "bidding" else 0.0,
        1.0 if phase == "trump" else 0.0,
        1.0 if phase == "kitty" else 0.0,
        1.0 if phase == "playing" else 0.0,
        1.0 if state.get("kitty_view_allowed") else 0.0,
        float(len(kitty)) / 6.0,
    ] + trump_one_hot + hand_counts + bid_values + seat_one_hot
    return EncodedState(tokens, scalars, mask, segments)


def card_to_action(card: str) -> int:
    return CARD_TO_ID[card] - 1


def action_to_card(action: int) -> str:
    return CARD_VOCAB[action]


def bid_to_action(bid: int | None) -> int:
    return BID_ACTION_OFFSET + BID_ACTIONS.index(bid)


def action_to_bid(action: int) -> int | None:
    return BID_ACTIONS[action - BID_ACTION_OFFSET]


def suit_to_action(suit: str) -> int:
    return SUIT_ACTION_OFFSET + SUIT_ACTIONS.index(suit)


def action_to_suit(action: int) -> str:
    return SUIT_ACTIONS[action - SUIT_ACTION_OFFSET]


def is_card_action(action: int) -> bool:
    return 0 <= action < len(CARD_VOCAB)


def is_bid_action(action: int) -> bool:
    return BID_ACTION_OFFSET <= action < SUIT_ACTION_OFFSET


def is_suit_action(action: int) -> bool:
    return SUIT_ACTION_OFFSET <= action < ACTION_SIZE
