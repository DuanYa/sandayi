from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Any

from sandayi.ai.features import action_to_bid, action_to_card, action_to_suit, bid_to_action, card_to_action, encode_state, is_bid_action, is_card_action, is_suit_action, suit_to_action
from sandayi.ai.strategy import RuleBasedStrategy
from sandayi.model.cards import effective_suit
from sandayi.model.game import Phase, RuleError, SandayiGame

logger = logging.getLogger(__name__)

PLAYER_IDS = [f"train_{idx}" for idx in range(4)]


@dataclass
class Transition:
    state: dict[str, Any]
    card_tokens: list[int]
    scalar_features: list[float]
    action_mask: list[float]
    segment_tokens: list[int]
    action: int
    player_id: str
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    is_banker: bool = False


class SandayiSelfPlayEnv:
    def __init__(self, seed: int | None = None, max_steps: int = 512) -> None:
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.rule_policy = RuleBasedStrategy()
        self.game = SandayiGame()
        self.steps = 0

    def reset(self) -> dict[str, Any]:
        self.game = SandayiGame()
        self.steps = 0
        for idx, pid in enumerate(PLAYER_IDS):
            self.game.add_player(pid, f"训练AI{idx + 1}", True)
        self.game.start()
        return self.current_view()

    def clone_state(self) -> SandayiGame:
        return copy.deepcopy(self.game)

    def restore_state(self, game: SandayiGame) -> None:
        self.game = game

    def current_player_id(self) -> str | None:
        view = self.game.public_view()
        return view.get("turn_player_id")

    def current_view(self) -> dict[str, Any]:
        current = self.current_player_id()
        return self.game.public_view(current) if current else self.game.public_view()

    def play_until_learning_turn(self, learner_id: str) -> dict[str, Any]:
        while not self.done() and self.current_player_id() != learner_id:
            self.step_rule()
        return self.current_view()

    def step_rule(self) -> None:
        pid = self.current_player_id()
        if not pid or self.done():
            return
        view = self.game.public_view(pid)
        action = self.rule_policy.decide(view)
        if action is None:
            raise RuleError(f"规则策略无法决策: {view.get('legal_actions')}")
        self.apply_action(pid, action)

    def step_learning_action(self, action_index: int) -> int | None:
        pid = self.current_player_id()
        if not pid:
            return None
        view = self.game.public_view(pid)
        legal = view.get("legal_actions", {})
        action = self._action_from_index(pid, action_index, legal)
        if action is None:
            fallback = self.rule_policy.decide(view)
            if fallback is None:
                raise RuleError(f"无合法训练动作: {legal}")
            action = self._sanitize_action(pid, fallback)
        if action.get("type") == "bury_select":
            fallback = self.rule_policy.decide(view)
            if fallback is None:
                raise RuleError(f"无合法扣底动作: {legal}")
            action = self._sanitize_action(pid, fallback)
        executed = self.apply_action(pid, action)
        return self._action_index_from_action(executed)

    def step_card_action(self, action_index: int) -> None:
        self.step_learning_action(action_index)

    def _action_from_index(self, player_id: str, action_index: int, legal: dict[str, Any]) -> dict[str, Any] | None:
        if legal.get("type") == "play" and is_card_action(action_index):
            card = action_to_card(action_index)
            if card in self._current_legal_cards(player_id):
                return {"type": "play", "card": card}
            return None
        if legal.get("type") == "bury" and is_card_action(action_index):
            card = action_to_card(action_index)
            if card in self.game.public_view(player_id).get("hand", []):
                return {"type": "bury_select", "card": card}
            return None
        if legal.get("type") == "bid" and is_bid_action(action_index):
            bid = action_to_bid(action_index)
            if bid is None or bid in legal.get("bids", []):
                return {"type": "bid", "bid": bid}
            return None
        if legal.get("type") == "trump" and is_suit_action(action_index):
            suit = action_to_suit(action_index)
            if suit in legal.get("suits", []):
                return {"type": "trump", "suit": suit}
            return None
        return None

    def _sanitize_action(self, player_id: str, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("type") != "play":
            return action
        legal_cards = self._current_legal_cards(player_id)
        if action.get("card") in legal_cards:
            return action
        if not legal_cards:
            raise RuleError(f"当前玩家没有合法可出牌: player={player_id} action={action}")
        logger.warning("训练动作被替换为实时合法牌 player=%s requested=%s legal=%s", player_id, action.get("card"), legal_cards)
        return {"type": "play", "card": legal_cards[0]}

    def _current_legal_cards(self, player_id: str) -> list[str]:
        try:
            player = self.game._player(player_id)
            if self.game.state.phase != Phase.PLAYING or self.game.public_view().get("turn_player_id") != player_id:
                return []
            trick = self.game.state.current_trick
            if trick is None or self.game.state.trump is None or not trick.plays:
                return [card.code for card in player.hand]
            legal: list[str] = []
            for card in player.hand:
                try:
                    self.game._validate_follow_rule(player, card, trick)
                except RuleError:
                    continue
                legal.append(card.code)
            return legal
        except Exception:
            logger.exception("读取实时合法牌失败 player=%s", player_id)
            return []

    def _action_index_from_action(self, action: dict[str, Any] | None) -> int | None:
        if not action:
            return None
        if action.get("type") in ("play", "bury_select") and action.get("card"):
            return card_to_action(action["card"])
        if action.get("type") == "bid":
            return bid_to_action(action.get("bid"))
        if action.get("type") == "trump":
            return suit_to_action(action.get("suit"))
        return None

    def apply_action(self, player_id: str, action: dict[str, Any]) -> dict[str, Any]:
        action = self._sanitize_action(player_id, action)
        action_type = action.get("type")
        if action_type == "bid":
            self.game.place_bid(player_id, action.get("bid"))
        elif action_type == "bury":
            self.game.bury_kitty(player_id, action.get("cards", []))
        elif action_type == "trump":
            self.game.choose_trump(player_id, action.get("suit"))
        elif action_type == "play":
            try:
                self.game.play_card(player_id, action.get("card"))
            except RuleError as exc:
                candidates = [card for card in self._current_legal_cards(player_id) if card != action.get("card")]
                last_error: Exception = exc
                for candidate in candidates:
                    fallback = {"type": "play", "card": candidate}
                    try:
                        self.game.play_card(player_id, candidate)
                        logger.warning("训练出牌触发规则错误，已逐张回退成功 player=%s action=%s fallback=%s error=%s", player_id, action, fallback, exc)
                        action = fallback
                        break
                    except RuleError as fallback_exc:
                        last_error = fallback_exc
                else:
                    logger.exception("训练出牌失败且所有手牌都不可回退 player=%s action=%s hand=%s last_error=%s", player_id, action, candidates, last_error)
                    raise last_error
        else:
            raise RuleError(f"未知训练动作: {action}")
        self.steps += 1
        return action

    def done(self) -> bool:
        return self.game.state.phase == Phase.FINISHED or self.steps >= self.max_steps

    def immediate_reward(self, player_id: str, before_points: tuple[int, int], before_tricks: int) -> float:
        after_tricks = len(self.game.state.completed_tricks)
        if after_tricks <= before_tricks:
            return 0.0
        banker_before, farmers_before = before_points
        banker_gain = self.game.state.banker_points - banker_before
        farmers_gain = self.game.state.farmers_points - farmers_before
        points = banker_gain + farmers_gain
        if points <= 0:
            return 0.02
        player_is_banker = player_id == self.game.state.banker_id
        own_side_gain = banker_gain if player_is_banker else farmers_gain
        reward = own_side_gain / 120.0
        return max(-0.3, min(0.3, reward))

    def final_rewards(self) -> dict[str, float]:
        if self.game.state.phase != Phase.FINISHED:
            return {pid: -0.2 for pid in PLAYER_IDS}
        rewards: dict[str, float] = {}
        result = self.game.state.result or {}
        loser_id = result.get("loser_player_id")
        banker_id = self.game.state.banker_id
        winner_side = self.game.state.winner_side
        multiplier = result.get("multiplier", 1)
        severity = 1.0 + 0.2 * (multiplier - 1)
        for player in self.game.state.players:
            is_banker = player.id == banker_id
            if loser_id:
                rewards[player.id] = -1.5 if player.id == loser_id else -0.3
            else:
                side = "banker" if is_banker else "farmers"
                won = winner_side == side or winner_side == player.id
                if is_banker:
                    rewards[player.id] = 1.5 * severity if won else -1.0 * severity
                else:
                    rewards[player.id] = 0.5 * severity if won else -0.5 * severity
        return rewards


def _select_masked_action(policy: Any, encoded: Any, device: str, epsilon: float = 0.0) -> tuple[int, float, float]:
    import torch
    from torch.distributions import Categorical

    legal_actions = [idx for idx, ok in enumerate(encoded.action_mask) if ok]
    if not legal_actions:
        raise RuleError("训练状态没有合法动作 mask")
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(legal_actions), 0.0, 0.0
    cards = torch.tensor([encoded.card_tokens], dtype=torch.long, device=device)
    scalars = torch.tensor([encoded.scalar_features], dtype=torch.float32, device=device)
    segments = torch.tensor([encoded.segment_tokens], dtype=torch.long, device=device)
    mask = torch.tensor(encoded.action_mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, value = policy(cards, scalars, segments)
        logits = logits[0].masked_fill(~mask, -1e4)
        dist = Categorical(logits=logits)
        action_idx = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action_idx, device=device)).item())
        value_item = float(value[0].item())
    return action_idx, log_prob, value_item




def _clone_encoded_with_mask(encoded: Any, mask: list[float]) -> Any:
    from dataclasses import replace

    return replace(encoded, action_mask=mask)


def _select_bury_cards(policy: Any, env: SandayiSelfPlayEnv, view: dict[str, Any], device: str, epsilon: float = 0.0) -> tuple[list[str], list[Transition]]:
    hand = list(view.get("hand", []))
    count = int(view.get("legal_actions", {}).get("count", 6))
    selected: list[str] = []
    transitions: list[Transition] = []
    base_encoded = encode_state(view)
    for _ in range(min(count, len(hand))):
        mask = [0.0] * len(base_encoded.action_mask)
        for card in hand:
            mask[card_to_action(card)] = 1.0
        encoded = _clone_encoded_with_mask(base_encoded, mask)
        action_idx, log_prob, value_item = _select_masked_action(policy, encoded, device, epsilon=epsilon)
        if not is_card_action(action_idx):
            action_idx = card_to_action(hand[0])
            log_prob = 0.0
            value_item = 0.0
        card = action_to_card(action_idx)
        if card not in hand:
            card = hand[0]
            action_idx = card_to_action(card)
            log_prob = 0.0
            value_item = 0.0
        selected.append(card)
        hand.remove(card)
        transitions.append(Transition(view, encoded.card_tokens, encoded.scalar_features, encoded.action_mask, encoded.segment_tokens, action_idx, env.current_player_id() or "", log_prob=log_prob, value=value_item))
    return selected, transitions

def _collect_episode(env: SandayiSelfPlayEnv, policy: Any, device: str, epsilon: float, with_log_prob: bool) -> list[Transition]:
    env.reset()
    transitions: list[Transition] = []
    policy.eval()
    while not env.done():
        pid = env.current_player_id()
        if not pid:
            break
        view = env.game.public_view(pid)
        legal = view.get("legal_actions", {})
        action_type = legal.get("type")
        encoded = encode_state(view)
        if not any(encoded.action_mask):
            action = env.rule_policy.decide(view)
            if action is None:
                break
            env.apply_action(pid, action)
            continue
        before_points = (env.game.state.banker_points, env.game.state.farmers_points)
        before_tricks = len(env.game.state.completed_tricks)
        if action_type == "bury":
            cards, bury_transitions = _select_bury_cards(policy, env, view, device, epsilon=epsilon)
            if not cards:
                action = env.rule_policy.decide(view)
                if action is None:
                    break
                env.apply_action(pid, action)
                continue
            env.apply_action(pid, {"type": "bury", "cards": cards})
            reward = env.immediate_reward(pid, before_points, before_tricks)
            for transition in bury_transitions:
                transition.player_id = pid
                transition.reward = reward
                if not with_log_prob:
                    transition.log_prob = 0.0
                    transition.value = 0.0
                transitions.append(transition)
            continue
        if action_type not in ("play", "bid", "trump"):
            action = env.rule_policy.decide(view)
            if action is None:
                break
            env.apply_action(pid, action)
            continue
        action_idx, log_prob, value_item = _select_masked_action(policy, encoded, device, epsilon=epsilon)
        executed_idx = env.step_learning_action(action_idx)
        if executed_idx is not None:
            transition = Transition(view, encoded.card_tokens, encoded.scalar_features, encoded.action_mask, encoded.segment_tokens, executed_idx, pid, log_prob=log_prob if with_log_prob else 0.0, value=value_item if with_log_prob else 0.0)
            transition.reward = env.immediate_reward(pid, before_points, before_tricks)
            transitions.append(transition)
    rewards = env.final_rewards()
    banker_id = env.game.state.banker_id
    for transition in transitions:
        transition.reward += rewards.get(transition.player_id, 0.0)
        transition.is_banker = transition.player_id == banker_id
    return transitions


def collect_deep_mc_episode(env: SandayiSelfPlayEnv, learner: Any, device: str, epsilon: float = 0.08) -> list[Transition]:
    return _collect_episode(env, learner, device, epsilon=epsilon, with_log_prob=False)


def collect_ppo_episode(env: SandayiSelfPlayEnv, policy: Any, device: str) -> list[Transition]:
    return _collect_episode(env, policy, device, epsilon=0.0, with_log_prob=True)
