from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sandayi Transformer AI with Deep Monte-Carlo self-play.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--save-path", default="models/sandayi_transformer_policy.pt")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
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

    replay_cards: list[list[int]] = []
    replay_scalars: list[list[float]] = []
    replay_masks: list[list[float]] = []
    replay_actions: list[int] = []
    replay_returns: list[float] = []
    max_replay = 50000

    for episode in range(1, args.episodes + 1):
        transitions = collect_deep_mc_episode(env, model, device=device, epsilon=args.epsilon)
        for item in transitions:
            replay_cards.append(item.card_tokens)
            replay_scalars.append(item.scalar_features)
            replay_masks.append(item.action_mask)
            replay_actions.append(item.action)
            replay_returns.append(item.reward)
        if len(replay_actions) > max_replay:
            overflow = len(replay_actions) - max_replay
            del replay_cards[:overflow]
            del replay_scalars[:overflow]
            del replay_masks[:overflow]
            del replay_actions[:overflow]
            del replay_returns[:overflow]
        if len(replay_actions) >= args.batch_size:
            cards = torch.tensor(replay_cards, dtype=torch.long)
            scalars = torch.tensor(replay_scalars, dtype=torch.float32)
            masks = torch.tensor(replay_masks, dtype=torch.bool)
            actions = torch.tensor(replay_actions, dtype=torch.long)
            returns = torch.tensor(replay_returns, dtype=torch.float32)
            dataset = TensorDataset(cards, scalars, masks, actions, returns)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
            model.train()
            total_loss = 0.0
            batches = 0
            for card_batch, scalar_batch, mask_batch, action_batch, return_batch in loader:
                card_batch = card_batch.to(device)
                scalar_batch = scalar_batch.to(device)
                mask_batch = mask_batch.to(device)
                action_batch = action_batch.to(device)
                return_batch = return_batch.to(device)
                valid_rows = mask_batch.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                if not bool(valid_rows.any().item()):
                    continue
                card_batch = card_batch[valid_rows]
                scalar_batch = scalar_batch[valid_rows]
                mask_batch = mask_batch[valid_rows]
                action_batch = action_batch[valid_rows]
                return_batch = return_batch[valid_rows].clamp(-4.0, 4.0)
                logits, values = model(card_batch, scalar_batch)
                logits = logits.masked_fill(~mask_batch, -1e4)
                log_probs = F.log_softmax(logits, dim=-1)
                chosen_log_probs = log_probs.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                advantage = (return_batch - values.detach()).clamp(-4.0, 4.0)
                policy_loss = -(chosen_log_probs * advantage).mean()
                value_loss = F.smooth_l1_loss(values, return_batch)
                probs = log_probs.exp().masked_fill(~mask_batch, 0.0)
                entropy = -(probs * log_probs.masked_fill(~mask_batch, 0.0)).sum(dim=-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1
        else:
            total_loss = 0.0
            batches = 0
        if episode % args.log_every == 0:
            avg_reward = sum(replay_returns[-max(1, len(transitions)):]) / max(1, len(transitions)) if transitions else 0.0
            print(f"episode={episode} transitions={len(transitions)} replay={len(replay_actions)} loss={total_loss / max(1, batches):.4f} last_avg_reward={avg_reward:.3f}")
        if episode % args.checkpoint_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "episode": episode, "algo": "deep_mc"}, checkpoint_path(episode))
        if episode % args.log_every == 0:
            torch.save({"model_state_dict": model.state_dict(), "episode": episode, "algo": "deep_mc"}, save_path)
    torch.save({"model_state_dict": model.state_dict(), "episode": args.episodes, "algo": "deep_mc"}, save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    main()
