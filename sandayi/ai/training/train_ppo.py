from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sandayi Transformer AI with PPO self-play.")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--save-path", default="models/sandayi_transformer_policy.pt")
    parser.add_argument("--checkpoint-every-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    import random

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from sandayi.ai.features import ACTION_SIZE, CARD_VOCAB, SCALAR_SIZE
    from sandayi.ai.network import create_policy_model
    from sandayi.ai.training.env import SandayiSelfPlayEnv, collect_ppo_episode

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

    def checkpoint_path(episodes: int) -> Path:
        return save_path.with_name(f"{save_path.stem}_episode_{episodes}{save_path.suffix}")

    last_checkpoint_episodes = 0

    for update in range(1, args.updates + 1):
        transitions = []
        for _ in range(args.episodes_per_update):
            transitions.extend(collect_ppo_episode(env, model, device=device))
        if not transitions:
            print(f"update={update} no transitions")
            continue
        cards = torch.tensor([t.card_tokens for t in transitions], dtype=torch.long)
        scalars = torch.tensor([t.scalar_features for t in transitions], dtype=torch.float32)
        masks = torch.tensor([t.action_mask for t in transitions], dtype=torch.bool)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
        old_values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
        returns = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        advantages = returns - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = TensorDataset(cards, scalars, masks, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model.train()
        total_loss = 0.0
        batches = 0
        for _ in range(args.epochs):
            for batch in loader:
                card_batch, scalar_batch, mask_batch, action_batch, old_log_prob_batch, return_batch, advantage_batch = [x.to(device) for x in batch]
                logits, values = model(card_batch, scalar_batch)
                valid_rows = mask_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                if not bool(valid_rows.any().item()):
                    continue
                logits = logits[valid_rows]
                values = values[valid_rows]
                mask_batch = mask_batch[valid_rows]
                action_batch = action_batch[valid_rows]
                old_log_prob_batch = old_log_prob_batch[valid_rows]
                return_batch = return_batch[valid_rows].clamp(-4.0, 4.0)
                advantage_batch = advantage_batch[valid_rows].clamp(-4.0, 4.0)
                logits = logits.masked_fill(~mask_batch, -1e4)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                new_log_probs = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                ratio = (new_log_probs - old_log_prob_batch).exp()
                unclipped = ratio * advantage_batch
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantage_batch
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.smooth_l1_loss(values, return_batch)
                entropy = -(probs * log_probs).masked_fill(~mask_batch, 0.0).sum(dim=-1).mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
        avg_reward = float(returns.mean().item())
        print(f"update={update} transitions={len(transitions)} loss={total_loss / max(1, batches):.4f} avg_reward={avg_reward:.3f}")
        completed_episodes = update * args.episodes_per_update
        while last_checkpoint_episodes + args.checkpoint_every_episodes <= completed_episodes:
            last_checkpoint_episodes += args.checkpoint_every_episodes
            torch.save({"model_state_dict": model.state_dict(), "update": update, "episodes": last_checkpoint_episodes, "algo": "ppo"}, checkpoint_path(last_checkpoint_episodes))
        if update % 10 == 0:
            torch.save({"model_state_dict": model.state_dict(), "update": update, "episodes": completed_episodes, "algo": "ppo"}, save_path)
    torch.save({"model_state_dict": model.state_dict(), "update": args.updates, "algo": "ppo"}, save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    main()
