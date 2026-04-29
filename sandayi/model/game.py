from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .cards import Card, Suit, card_sort_key, effective_suit, full_deck

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    WAITING = "waiting"
    BIDDING = "bidding"
    KITTY = "kitty"
    TRUMP = "trump"
    PLAYING = "playing"
    FINISHED = "finished"


BID_OPTIONS = [60, 65, 70, 75, 80]
BASE_SCORE = {60: 1, 65: 2, 70: 3, 75: 4, 80: 5}


@dataclass
class PlayerState:
    id: str
    name: str
    is_ai: bool = False
    hand: list[Card] = field(default_factory=list)
    score_delta: int = 0
    online: bool = True
    passed_bid: bool = False


@dataclass
class BidRecord:
    player_id: str
    bid: Optional[int]


@dataclass
class TrickPlay:
    player_id: str
    card: Card
    order: int


@dataclass
class Trick:
    leader_id: str
    plays: list[TrickPlay] = field(default_factory=list)


@dataclass
class GameState:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    players: list[PlayerState] = field(default_factory=list)
    phase: Phase = Phase.WAITING
    turn_index: int = 0
    bid_start_index: int = 0
    bid_records: list[BidRecord] = field(default_factory=list)
    highest_bid: Optional[int] = None
    banker_id: Optional[str] = None
    kitty: list[Card] = field(default_factory=list)
    trump: Optional[Suit] = None
    current_trick: Optional[Trick] = None
    completed_tricks: list[Trick] = field(default_factory=list)
    banker_points: int = 0
    farmers_points: int = 0
    kitty_view_allowed: bool = False
    winner_side: Optional[str] = None
    result: dict[str, Any] = field(default_factory=dict)
    version: int = 0


class RuleError(ValueError):
    pass


class SandayiGame:
    def __init__(self) -> None:
        self.state = GameState()

    def add_player(self, player_id: str, name: str, is_ai: bool = False) -> None:
        if self.state.phase != Phase.WAITING:
            raise RuleError("游戏已开始，不能加入")
        if len(self.state.players) >= 4:
            raise RuleError("房间人数已满")
        if any(p.id == player_id for p in self.state.players):
            raise RuleError("玩家已存在")
        self.state.players.append(PlayerState(player_id, name, is_ai))
        self._touch()

    def start(self) -> None:
        if len(self.state.players) != 4:
            raise RuleError("必须四名玩家才能开始")
        deck = full_deck()
        random.shuffle(deck)
        for player in self.state.players:
            player.hand.clear()
            player.passed_bid = False
        for i in range(48):
            self.state.players[i % 4].hand.append(deck[i])
        self.state.kitty = deck[48:]
        self.state.phase = Phase.BIDDING
        self.state.bid_records.clear()
        self.state.highest_bid = None
        self.state.banker_id = None
        self.state.trump = None
        self.state.banker_points = 0
        self.state.farmers_points = 0
        self.state.completed_tricks.clear()
        self.state.current_trick = None
        self.state.result.clear()
        self.state.winner_side = None
        self.state.bid_start_index = self._find_diamond_three_index()
        self.state.turn_index = self.state.bid_start_index
        self._sort_all_hands()
        self._touch()
        logger.info("游戏开始 game=%s version=%s bid_start=%s players=%s", self.state.id, self.state.version, self._current_player().id, [p.id for p in self.state.players])

    def place_bid(self, player_id: str, bid: Optional[int]) -> None:
        self._require_phase(Phase.BIDDING)
        player = self._current_player()
        if player.id != player_id:
            raise RuleError("未轮到该玩家叫牌")
        if bid is None:
            player.passed_bid = True
            self.state.bid_records.append(BidRecord(player_id, None))
            logger.info("玩家放弃叫牌 game=%s player=%s records=%s", self.state.id, player_id, len(self.state.bid_records))
        else:
            self._validate_bid(bid)
            self.state.highest_bid = bid
            self.state.banker_id = player_id
            self.state.bid_records.append(BidRecord(player_id, bid))
            logger.info("玩家叫牌 game=%s player=%s bid=%s records=%s", self.state.id, player_id, bid, len(self.state.bid_records))
            if bid == 80:
                self._finish_bidding()
                self._touch()
                return
        if self._bidding_complete():
            self._finish_bidding()
        else:
            self._advance_turn_to_next_bidder()
        self._touch()

    def bury_kitty(self, player_id: str, card_codes: list[str]) -> None:
        self._require_phase(Phase.KITTY)
        if player_id != self.state.banker_id:
            raise RuleError("只有庄家可以扣底")
        if len(card_codes) != 6:
            raise RuleError("必须扣 6 张底牌")
        player = self._player(player_id)
        cards = self._take_cards(player, card_codes)
        self.state.kitty = cards
        self.state.phase = Phase.PLAYING
        self.state.turn_index = self._player_index(player_id)
        self.state.current_trick = Trick(player_id)
        self._sort_all_hands()
        self._touch()
        logger.info("庄家扣底 game=%s version=%s player=%s buried=%s", self.state.id, self.state.version, player_id, card_codes)

    def choose_trump(self, player_id: str, suit_value: str) -> None:
        self._require_phase(Phase.TRUMP)
        if player_id != self.state.banker_id:
            raise RuleError("只有庄家可以定主")
        suit = Suit(suit_value)
        if suit == Suit.JOKER:
            raise RuleError("主牌花色只能是四种普通花色")
        self.state.trump = suit
        self.state.phase = Phase.KITTY
        self._sort_all_hands()
        self._touch()
        logger.info("庄家定主 game=%s version=%s player=%s trump=%s", self.state.id, self.state.version, player_id, suit_value)

    def play_card(self, player_id: str, card_code: str) -> None:
        self._require_phase(Phase.PLAYING)
        if self._current_player().id != player_id:
            raise RuleError("未轮到该玩家出牌")
        player = self._player(player_id)
        card = self._card_from_hand(player, card_code)
        if card is None:
            logger.error("出牌失败，手牌中没有该牌 game=%s player=%s card=%s hand=%s legal=%s", self.state.id, player_id, card_code, [c.code for c in player.hand], self.legal_actions(player_id))
            raise RuleError("手牌中没有这张牌")
        trick = self.state.current_trick
        if trick is None or self.state.trump is None:
            raise RuleError("当前轮次不存在")
        self._validate_follow_rule(player, card, trick)
        player.hand.remove(card)
        trick.plays.append(TrickPlay(player_id, card, len(trick.plays)))
        logger.info("玩家出牌 game=%s player=%s card=%s trick_count=%s hand_left=%s", self.state.id, player_id, card_code, len(trick.plays), len(player.hand))
        if len(trick.plays) == 4:
            self._finish_trick(trick)
        else:
            self.state.turn_index = (self.state.turn_index + 1) % 4
        self._touch()

    def legal_actions(self, player_id: str) -> dict[str, Any]:
        player = self._player(player_id)
        if self.state.phase == Phase.BIDDING and self._current_player().id == player_id:
            min_bid = self._min_legal_bid()
            bids = [b for b in BID_OPTIONS if b >= min_bid and (self.state.highest_bid is None or b > self.state.highest_bid)]
            return {"type": "bid", "bids": bids, "can_pass": True}
        if self.state.phase == Phase.KITTY and player_id == self.state.banker_id:
            return {"type": "bury", "count": 6}
        if self.state.phase == Phase.TRUMP and player_id == self.state.banker_id:
            return {"type": "trump", "suits": ["S", "H", "C", "D"]}
        if self.state.phase == Phase.PLAYING and self._current_player().id == player_id:
            hand_codes = {card.code for card in player.hand}
            cards = [c.code for c in self._legal_play_cards(player) if c.code in hand_codes]
            return {"type": "play", "cards": cards}
        return {"type": "wait"}

    def public_view(self, viewer_id: Optional[str] = None) -> dict[str, Any]:
        has_viewer = viewer_id and any(p.id == viewer_id for p in self.state.players)
        viewer = self._player(viewer_id) if has_viewer else None
        return {
            "id": self.state.id,
            "version": self.state.version,
            "phase": self.state.phase.value,
            "viewer_id": viewer_id,
            "players": [{"id": p.id, "name": p.name, "is_ai": p.is_ai, "hand_count": len(p.hand), "online": p.online, "score_delta": p.score_delta} for p in self.state.players],
            "turn_player_id": self._current_player().id if self.state.players and self.state.phase not in (Phase.WAITING, Phase.FINISHED) else None,
            "banker_id": self.state.banker_id,
            "highest_bid": self.state.highest_bid,
            "bid_records": [r.__dict__ for r in self.state.bid_records],
            "trump": self.state.trump.value if self.state.trump else None,
            "banker_points": self.state.banker_points,
            "farmers_points": self.state.farmers_points,
            "current_trick": self._trick_view(self.state.current_trick),
            "last_trick": self._trick_view(self.state.completed_tricks[-1]) if self.state.completed_tricks else None,
            "completed_tricks": len(self.state.completed_tricks),
            "completed_trick_plays": [[{"player_id": play.player_id, "card": play.card.code, "order": play.order} for play in trick.plays] for trick in self.state.completed_tricks],
            "kitty_view_allowed": self.state.kitty_view_allowed,
            "result": self.state.result,
            "hand": [c.code for c in viewer.hand] if viewer else [],
            "hand_labels": [c.label for c in viewer.hand] if viewer else [],
            "kitty": [c.code for c in self.state.kitty] if self._can_view_kitty(viewer_id) else [],
            "legal_actions": self.legal_actions(viewer_id) if has_viewer else {"type": "wait"},
        }

    def _can_view_kitty(self, viewer_id: Optional[str]) -> bool:
        if not viewer_id:
            return False
        if viewer_id == self.state.banker_id:
            return True
        if self.state.phase == Phase.BIDDING:
            return False
        return bool(self.state.kitty_view_allowed)
    def _finish_bidding(self) -> None:
        if self.state.highest_bid is None or self.state.banker_id is None:
            loser = self._common_trump_loser()
            trump_count = self._common_trump_count(loser)
            penalty_bid = self._pass_penalty_bid(trump_count)
            base = BASE_SCORE[penalty_bid]
            for player in self.state.players:
                player.score_delta += -3 * base if player.id == loser.id else base
            self.state.winner_side = "farmers"
            self.state.result = {
                "winner_side": "farmers",
                "reason": f"全员放弃，常主 {trump_count} 张者判负，按 {penalty_bid} 破牌结算",
                "result_type": "全放弃破牌",
                "bid": penalty_bid,
                "base": base,
                "multiplier": 1,
                "banker_delta": -3 * base,
                "farmer_delta": base,
                "farmers_points": 100 - penalty_bid,
                "banker_points": penalty_bid,
                "farmers_need": 100 - penalty_bid,
                "common_trump_count": trump_count,
                "loser_player_id": loser.id,
                "is_cheng_pai": False,
                "is_po_pai": True,
                "is_guang_pai": False,
                "is_kou_di": False,
                "is_lian_kou_dai_po": False,
                "is_ping_fen": False,
            }
            self.state.phase = Phase.FINISHED
            return
        banker = self._player(self.state.banker_id)
        banker.hand.extend(self.state.kitty)
        self.state.kitty_view_allowed = self.state.highest_bid is not None and self.state.highest_bid <= 65
        self.state.phase = Phase.TRUMP
        self.state.turn_index = self._player_index(banker.id)
        self._sort_all_hands()

    def _finish_trick(self, trick: Trick) -> None:
        winner_play = self._trick_winner(trick)
        points = sum(play.card.points for play in trick.plays)
        logger.info("轮次结束 game=%s winner=%s points=%s plays=%s", self.state.id, winner_play.player_id, points, [(p.player_id, p.card.code) for p in trick.plays])
        if winner_play.player_id == self.state.banker_id:
            self.state.banker_points += points
        else:
            self.state.farmers_points += points
        self.state.completed_tricks.append(trick)
        if all(len(p.hand) == 0 for p in self.state.players):
            self._finish_game(winner_play.player_id != self.state.banker_id)
            return
        self.state.turn_index = self._player_index(winner_play.player_id)
        self.state.current_trick = Trick(winner_play.player_id)

    def _finish_game(self, last_won_by_farmer: bool) -> None:
        bid = self.state.highest_bid or 60
        base = BASE_SCORE[bid]
        farmers_need = 100 - bid
        farmers_points = self.state.farmers_points
        banker_points = self.state.banker_points
        kou_di = last_won_by_farmer
        po_pai = farmers_points >= farmers_need
        shang_che = farmers_points > farmers_need * 1.5
        guang_pai = farmers_points == 0
        lian_kou_dai_po = kou_di and po_pai
        farmer_win = kou_di or po_pai

        if farmer_win:
            result_type = "破牌"
            reason = "闲家破牌胜利"
            multiplier = 1
            if shang_che and not kou_di:
                multiplier = 2
                result_type = "上车"
                reason = "闲家上车胜利"
            if kou_di:
                multiplier = 2
                result_type = "抠底"
                reason = "闲家抠底胜利"
            if lian_kou_dai_po:
                multiplier = 4
                result_type = "连抠带破"
                reason = "闲家连抠带破胜利"
        else:
            multiplier = 3 if guang_pai else 1
            result_type = "光牌" if guang_pai else "成牌"
            reason = "庄家光牌胜利" if guang_pai else "庄家成牌胜利"

        banker_delta = 3 * base * multiplier * (-1 if farmer_win else 1)
        farmer_delta = base * multiplier * (1 if farmer_win else -1)
        for player in self.state.players:
            player.score_delta += banker_delta if player.id == self.state.banker_id else farmer_delta
        self.state.winner_side = "farmers" if farmer_win else "banker"
        self.state.result = {
            "winner_side": self.state.winner_side,
            "reason": reason,
            "result_type": result_type,
            "bid": bid,
            "base": base,
            "multiplier": multiplier,
            "banker_delta": banker_delta,
            "farmer_delta": farmer_delta,
            "farmers_points": farmers_points,
            "banker_points": banker_points,
            "farmers_need": farmers_need,
            "shang_che_threshold": farmers_need * 1.5,
            "is_cheng_pai": not farmer_win,
            "is_po_pai": po_pai,
            "is_shang_che": shang_che and farmer_win,
            "is_guang_pai": guang_pai,
            "is_kou_di": kou_di,
            "is_lian_kou_dai_po": lian_kou_dai_po,
        }
        self.state.phase = Phase.FINISHED
    def _validate_bid(self, bid: int) -> None:
        if bid not in BID_OPTIONS:
            raise RuleError("叫分只能是 60、65、70、75、80")
        if self.state.highest_bid is not None and bid <= self.state.highest_bid:
            raise RuleError("叫分必须高于当前最高叫分")
        if bid < self._min_legal_bid():
            raise RuleError("叫分低于当前起叫限制")

    def _min_legal_bid(self) -> int:
        return 60

    def _bidding_complete(self) -> bool:
        if self.state.highest_bid is None:
            return len(self.state.bid_records) >= 4
        acted_players = {record.player_id for record in self.state.bid_records}
        return len(acted_players) >= 4

    def _advance_turn_to_next_bidder(self) -> None:
        for step in range(1, 5):
            idx = (self.state.turn_index + step) % 4
            if not self.state.players[idx].passed_bid:
                self.state.turn_index = idx
                return

    def _validate_follow_rule(self, player: PlayerState, card: Card, trick: Trick) -> None:
        if not trick.plays:
            return
        trump = self.state.trump
        assert trump is not None
        # 王、2 和主花色统一视为 TRUMP；副牌按原花色跟牌。
        lead_suit = effective_suit(trick.plays[0].card, trump)
        remaining_hand = [hand_card for hand_card in player.hand if hand_card != card]
        has_lead = any(effective_suit(c, trump) == lead_suit for c in remaining_hand)
        if effective_suit(card, trump) != lead_suit and has_lead:
            raise RuleError("必须优先跟出首家同花色牌")

    def _legal_play_cards(self, player: PlayerState) -> list[Card]:
        trick = self.state.current_trick
        if trick is None or not trick.plays or self.state.trump is None:
            return list(player.hand)
        legal: list[Card] = []
        for card in player.hand:
            try:
                self._validate_follow_rule(player, card, trick)
            except RuleError:
                continue
            legal.append(card)
        return legal
    def _trick_winner(self, trick: Trick) -> TrickPlay:
        trump = self.state.trump
        assert trump is not None
        lead_suit = effective_suit(trick.plays[0].card, trump)
        best = trick.plays[0]
        for play in trick.plays[1:]:
            if self._beats(play.card, best.card, lead_suit):
                best = play
        return best

    def _beats(self, challenger: Card, current: Card, lead_suit: str) -> bool:
        trump = self.state.trump
        assert trump is not None
        c_suit = effective_suit(challenger, trump)
        b_suit = effective_suit(current, trump)
        if c_suit == "TRUMP" and b_suit != "TRUMP":
            return True
        if c_suit != "TRUMP" and b_suit == "TRUMP":
            return False
        if c_suit == b_suit and c_suit == "TRUMP":
            return card_sort_key(challenger, trump) > card_sort_key(current, trump)
        if c_suit == b_suit == lead_suit:
            return card_sort_key(challenger, trump) > card_sort_key(current, trump)
        return False

    def _find_diamond_three_index(self) -> int:
        target = Card(Suit.DIAMOND, "3")
        for i, player in enumerate(self.state.players):
            if target in player.hand:
                return i
        return 0

    def _common_trump_loser(self) -> PlayerState:
        return max(enumerate(self.state.players), key=lambda item: (self._common_trump_count(item[1]), self._highest_common_trump_value(item[1]), -item[0]))[1]

    def _common_trump_count(self, player: PlayerState) -> int:
        return sum(1 for card in player.hand if card.suit == Suit.JOKER or card.rank == "2")

    def _highest_common_trump_value(self, player: PlayerState) -> int:
        values = [self._common_trump_value(card) for card in player.hand if card.suit == Suit.JOKER or card.rank == "2"]
        return max(values, default=0)

    def _common_trump_value(self, card: Card) -> int:
        if card.rank == "BJ":
            return 7
        if card.rank == "XJ":
            return 6
        if card.rank == "2":
            return 5
        return 0

    def _pass_penalty_bid(self, common_trump_count: int) -> int:
        if common_trump_count <= 2:
            return 60
        if common_trump_count == 3:
            return 65
        if common_trump_count == 4:
            return 70
        if common_trump_count == 5:
            return 75
        return 80
    def _take_cards(self, player: PlayerState, card_codes: list[str]) -> list[Card]:
        temp = list(player.hand)
        taken: list[Card] = []
        for code in card_codes:
            card = self._card_from_cards(temp, code)
            if card is None:
                logger.error("扣底失败，手牌中没有该牌 player=%s card=%s hand=%s", player.id, code, [c.code for c in temp])
                raise RuleError("扣底牌不在手牌中")
            temp.remove(card)
            taken.append(card)
        player.hand = temp
        return taken

    def _card_from_hand(self, player: PlayerState, card_code: str) -> Optional[Card]:
        return self._card_from_cards(player.hand, card_code)

    def _card_from_cards(self, cards: list[Card], card_code: str) -> Optional[Card]:
        for card in cards:
            if card.code == card_code:
                return card
        return None
    def _player(self, player_id: str) -> PlayerState:
        for player in self.state.players:
            if player.id == player_id:
                return player
        raise RuleError("玩家不存在")

    def _player_index(self, player_id: str) -> int:
        for i, player in enumerate(self.state.players):
            if player.id == player_id:
                return i
        raise RuleError("玩家不存在")

    def _current_player(self) -> PlayerState:
        return self.state.players[self.state.turn_index]

    def _require_phase(self, phase: Phase) -> None:
        if self.state.phase != phase:
            raise RuleError(f"当前阶段不能执行该操作: {self.state.phase.value}")

    def _sort_all_hands(self) -> None:
        for player in self.state.players:
            player.hand.sort(key=lambda c: card_sort_key(c, self.state.trump), reverse=True)

    def _touch(self) -> None:
        self.state.version += 1

    def _trick_view(self, trick: Optional[Trick]) -> Optional[dict[str, Any]]:
        if trick is None:
            return None
        return {"leader_id": trick.leader_id, "plays": [{"player_id": p.player_id, "card": p.card.code, "label": p.card.label} for p in trick.plays]}
