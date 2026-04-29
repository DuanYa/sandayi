from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Suit(str, Enum):
    SPADE = "S"
    HEART = "H"
    CLUB = "C"
    DIAMOND = "D"
    JOKER = "J"


SUIT_NAMES = {"S": "黑桃", "H": "红桃", "C": "梅花", "D": "方块", "J": "王"}
RANK_NAMES = {"3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "10": "10", "J": "J", "Q": "Q", "K": "K", "A": "A", "2": "2", "XJ": "小王", "BJ": "大王"}
RANK_ORDER = {rank: idx for idx, rank in enumerate(["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])}
POINTS = {"5": 5, "10": 10, "K": 10}


@dataclass(frozen=True, order=False)
class Card:
    suit: Suit
    rank: str

    @property
    def code(self) -> str:
        if self.suit == Suit.JOKER:
            return self.rank
        return f"{self.suit.value}{self.rank}"

    @property
    def points(self) -> int:
        return POINTS.get(self.rank, 0)

    @property
    def label(self) -> str:
        if self.suit == Suit.JOKER:
            return RANK_NAMES[self.rank]
        return f"{SUIT_NAMES[self.suit.value]}{RANK_NAMES[self.rank]}"

    @staticmethod
    def from_code(code: str) -> "Card":
        if code in ("XJ", "BJ"):
            return Card(Suit.JOKER, code)
        suit = Suit(code[0])
        rank = code[1:]
        if rank not in RANK_ORDER and rank != "2":
            raise ValueError(f"非法牌点: {code}")
        return Card(suit, rank)


def full_deck() -> list[Card]:
    cards: list[Card] = []
    for suit in (Suit.SPADE, Suit.HEART, Suit.CLUB, Suit.DIAMOND):
        for rank in ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]:
            cards.append(Card(suit, rank))
    cards.extend([Card(Suit.JOKER, "XJ"), Card(Suit.JOKER, "BJ")])
    return cards


def card_sort_key(card: Card, trump: Optional[Suit] = None) -> tuple[int, int, int, str]:
    if card.rank == "BJ":
        return (7, 0, 0, card.code)
    if card.rank == "XJ":
        return (6, 0, 0, card.code)
    if card.rank == "2" and trump and card.suit == trump:
        return (5, 4, 0, card.code)
    if card.rank == "2":
        return (4, _suit_group(card.suit, trump), 0, card.code)
    if trump and card.suit == trump:
        return (3, 0, RANK_ORDER[card.rank], card.code)
    return (2, _suit_group(card.suit, trump), RANK_ORDER[card.rank], card.code)

def _suit_group(suit: Suit, trump: Optional[Suit]) -> int:
    base_order = [Suit.SPADE, Suit.HEART, Suit.CLUB, Suit.DIAMOND]
    if trump and trump in base_order:
        ordered = [trump] + [item for item in base_order if item != trump]
    else:
        ordered = base_order
    # reverse=True 排序，因此越靠前的花色分组值越大。
    return len(ordered) - ordered.index(suit)


def is_trump(card: Card, trump: Suit) -> bool:
    return card.suit == Suit.JOKER or card.rank == "2" or card.suit == trump


def effective_suit(card: Card, trump: Suit) -> str:
    return "TRUMP" if is_trump(card, trump) else card.suit.value
