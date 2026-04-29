from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sandayi Transformer AI with on-policy Monte-Carlo actor-critic self-play.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--save-path", default="models/sandayi_transformer_policy.pt")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    import random

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from sandayi.ai.features import ACTION_SIZE, CARD_VOCAB, SCALAR_SIZE
    from sandayi.ai.network import create_policy_model
    from sandayi.ai.training.env import SandayiSelfPlayEnv, collect_deep_mc_episode

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_policy_model(nn, vocab_size=len(CARD_VOCAB) + 1, action_size=ACTION_SIZE, scalar_size=SCALAR_SIZE).to(device)
    print(f"device={device} cuda_available={torch.cuda.is_available()} parameters={model.parameter_count():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    env = SandayiSelfPlayEnv(seed=args.seed)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(step: int) -> Path:
        return save_path.with_name(f"{save_path.stem}_episode_{step}{save_path.suffix}")

    # --- log field abbreviations ---
    # ep:episode tr:transitions atr:avg_transitions l:loss p:policy_loss v:value_loss
    # ent:entropy r:avg_reward rb:avg_banker_reward rf:avg_farmer_reward
    # bid:avg_bid bp:avg_banker_points fp:avg_farmer_points
    # bw:banker_win_rate fw:farmer_win_rate
    # pa:pass_all_rate cp:成牌_rate pp:破牌_rate sc:上车_rate kd:抠底_rate lk:连抠带破_rate gp:光牌_rate
    # bt:avg_banker_transitions ft:avg_farmer_transitions
    # mp:avg_max_action_prob gn:avg_grad_norm vp:avg_value_prediction vt:avg_value_target

    W = args.log_every
    recent_transitions: list[int] = []
    recent_rewards: list[float] = []
    recent_banker_rewards: list[float] = []
    recent_farmer_rewards: list[float] = []
    recent_bids: list[int] = []
    recent_banker_points: list[int] = []
    recent_farmer_points: list[int] = []
    recent_banker_transitions: list[int] = []
    recent_farmer_transitions: list[int] = []
    win_counts = {"banker": 0, "farmer": 0}
    result_counts = {"cp": 0, "pp": 0, "sc": 0, "kd": 0, "lk": 0, "gp": 0, "pa": 0}
    recent_finished = 0
    last_metrics: dict[str, float] = {
        "loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0,
        "max_prob": 0.0, "grad_norm": 0.0, "value_pred": 0.0, "value_target": 0.0,
    }

    for episode in range(1, args.episodes + 1):
        transitions = collect_deep_mc_episode(env, model, device=device, epsilon=args.epsilon)

        banker_id = env.game.state.banker_id
        banker_t = [t for t in transitions if t.player_id == banker_id]
        farmer_t = [t for t in transitions if t.player_id != banker_id]

        recent_transitions.append(len(transitions))
        recent_rewards.extend([t.reward for t in transitions])
        recent_banker_rewards.extend([t.reward for t in banker_t])
        recent_farmer_rewards.extend([t.reward for t in farmer_t])
        recent_banker_transitions.append(len(banker_t))
        recent_farmer_transitions.append(len(farmer_t))
        for lst in [recent_transitions, recent_banker_transitions, recent_farmer_transitions]:
            while len(lst) > W:
                lst.pop(0)
        for lst in [recent_rewards, recent_banker_rewards, recent_farmer_rewards]:
            while len(lst) > 10000:
                lst.pop(0)

        result = env.game.state.result or {}
        if result:
            recent_finished += 1
            recent_banker_points.append(int(result.get("banker_points", env.game.state.banker_points)))
            recent_farmer_points.append(int(result.get("farmers_points", env.game.state.farmers_points)))
            while len(recent_banker_points) > W:
                recent_banker_points.pop(0)
            while len(recent_farmer_points) > W:
                recent_farmer_points.pop(0)
            ws = result.get("winner_side", "")
            if ws == "banker":
                win_counts["banker"] += 1
            elif ws == "farmers":
                win_counts["farmer"] += 1
            rt = result.get("result_type", "")
            if rt == "全放弃破牌":
                result_counts["pa"] += 1
            elif rt == "成牌":
                result_counts["cp"] += 1
            elif rt == "破牌":
                result_counts["pp"] += 1
            elif rt == "上车":
                result_counts["sc"] += 1
            elif rt == "抠底":
                result_counts["kd"] += 1
            elif rt == "连抠带破":
                result_counts["lk"] += 1
            elif rt == "光牌":
                result_counts["gp"] += 1
        if env.game.state.highest_bid:
            recent_bids.append(int(env.game.state.highest_bid))
            while len(recent_bids) > W:
                recent_bids.pop(0)

        if len(transitions) >= 1:
            cards = torch.tensor([t.card_tokens for t in transitions], dtype=torch.long)
            scalars = torch.tensor([t.scalar_features for t in transitions], dtype=torch.float32)
            masks = torch.tensor([t.action_mask for t in transitions], dtype=torch.bool)
            segments = torch.tensor([t.segment_tokens for t in transitions], dtype=torch.long)
            actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
            returns = torch.tensor([t.reward for t in transitions], dtype=torch.float32).clamp(-4.0, 4.0)
            weights = torch.tensor([3.0 if t.is_banker else 1.0 for t in transitions], dtype=torch.float32)
            dataset = TensorDataset(cards, scalars, masks, segments, actions, returns, weights)
            loader = DataLoader(dataset, batch_size=min(args.batch_size, len(transitions)), shuffle=True, drop_last=False)
            model.train()
            total_loss = 0.0
            total_policy = 0.0
            total_value = 0.0
            total_entropy = 0.0
            total_max_prob = 0.0
            total_grad_norm = 0.0
            total_value_pred = 0.0
            total_value_target = 0.0
            batches = 0
            for _ in range(args.update_epochs):
                for card_batch, scalar_batch, mask_batch, segment_batch, action_batch, return_batch, weight_batch in loader:
                    card_batch = card_batch.to(device)
                    scalar_batch = scalar_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    segment_batch = segment_batch.to(device)
                    action_batch = action_batch.to(device)
                    return_batch = return_batch.to(device)
                    weight_batch = weight_batch.to(device)
                    valid_rows = mask_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    if not bool(valid_rows.any().item()):
                        continue
                    card_batch = card_batch[valid_rows]
                    scalar_batch = scalar_batch[valid_rows]
                    mask_batch = mask_batch[valid_rows]
                    segment_batch = segment_batch[valid_rows]
                    action_batch = action_batch[valid_rows]
                    return_batch = return_batch[valid_rows]
                    weight_batch = weight_batch[valid_rows]
                    logits, values = model(card_batch, scalar_batch, segment_batch)
                    logits = logits.masked_fill(~mask_batch, -1e4)
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().masked_fill(~mask_batch, 0.0)
                    chosen_log_probs = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1).clamp(min=-20.0, max=0.0)
                    advantage = (return_batch - values.detach()).clamp(-4.0, 4.0)
                    if advantage.numel() > 1:
                        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
                        advantage = advantage.clamp(-4.0, 4.0)
                    policy_loss = -(chosen_log_probs * advantage * weight_batch).sum() / weight_batch.sum()
                    value_loss = (weight_batch * F.smooth_l1_loss(values, return_batch, reduction="none")).sum() / weight_batch.sum()
                    entropy = -(probs * log_probs.masked_fill(~mask_batch, 0.0)).sum(dim=-1).mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                    optimizer.step()
                    total_loss += float(loss.item())
                    total_policy += float(policy_loss.item())
                    total_value += float(value_loss.item())
                    total_entropy += float(entropy.item())
                    total_max_prob += float(probs.max(dim=-1).values.mean().item())
                    total_grad_norm += gn
                    total_value_pred += float(values.mean().item())
                    total_value_target += float(return_batch.mean().item())
                    batches += 1
            nb = max(1, batches)
            last_metrics = {
                "loss": total_loss / nb,
                "policy": total_policy / nb,
                "value": total_value / nb,
                "entropy": total_entropy / nb,
                "max_prob": total_max_prob / nb,
                "grad_norm": total_grad_norm / nb,
                "value_pred": total_value_pred / nb,
                "value_target": total_value_target / nb,
            }

        if episode % args.log_every == 0:
            n_tr = max(1, sum(recent_transitions))
            atr = sum(recent_transitions) / max(1, len(recent_transitions))
            r = sum(recent_rewards[-n_tr:]) / n_tr
            rb = sum(recent_banker_rewards[-n_tr:]) / max(1, len(recent_banker_rewards[-n_tr:]))
            rf = sum(recent_farmer_rewards[-n_tr:]) / max(1, len(recent_farmer_rewards[-n_tr:]))
            bid_v = sum(recent_bids) / max(1, len(recent_bids))
            bp_v = sum(recent_banker_points) / max(1, len(recent_banker_points))
            fp_v = sum(recent_farmer_points) / max(1, len(recent_farmer_points))
            bt_v = sum(recent_banker_transitions) / max(1, len(recent_banker_transitions))
            ft_v = sum(recent_farmer_transitions) / max(1, len(recent_farmer_transitions))
            nf = max(1, recent_finished)
            bw = win_counts["banker"] / nf
            fw = win_counts["farmer"] / nf
            pa = result_counts["pa"] / nf
            cp = result_counts["cp"] / nf
            pp = result_counts["pp"] / nf
            sc = result_counts["sc"] / nf
            kd = result_counts["kd"] / nf
            lk = result_counts["lk"] / nf
            gp = result_counts["gp"] / nf
            m = last_metrics
            print(
                f"ep={episode} tr={len(transitions)} atr={atr:.1f} "
                f"l={m['loss']:.4f} p={m['policy']:.4f} v={m['value']:.4f} ent={m['entropy']:.4f} "
                f"r={r:.3f} rb={rb:.3f} rf={rf:.3f} "
                f"bid={bid_v:.1f} bp={bp_v:.1f} fp={fp_v:.1f} "
                f"bw={bw:.3f} fw={fw:.3f} "
                f"pa={pa:.3f} cp={cp:.3f} pp={pp:.3f} sc={sc:.3f} kd={kd:.3f} lk={lk:.3f} gp={gp:.3f} "
                f"bt={bt_v:.1f} ft={ft_v:.1f} "
                f"mp={m['max_prob']:.3f} gn={m['grad_norm']:.4f} vp={m['value_pred']:.3f} vt={m['value_target']:.3f}"
            )
            recent_finished = 0
            win_counts = {"banker": 0, "farmer": 0}
            result_counts = {"cp": 0, "pp": 0, "sc": 0, "kd": 0, "lk": 0, "gp": 0, "pa": 0}
            torch.save({"model_state_dict": model.state_dict(), "episode": episode, "algo": "deep_mc"}, save_path)
        if episode % args.checkpoint_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "episode": episode, "algo": "deep_mc"}, checkpoint_path(episode))
    torch.save({"model_state_dict": model.state_dict(), "episode": args.episodes, "algo": "deep_mc"}, save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    main()
