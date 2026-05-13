import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PHASES = ["approach", "descend", "close", "lift", "transfer", "place", "open", "retreat"]


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a state-based behavior cloning policy.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_phase", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.dataset, map_location="cpu")
    obs = data["obs"].float()
    actions = data["actions"].float()
    if args.use_phase:
        phase_to_index = {name: index for index, name in enumerate(PHASES)}
        phase_indices = torch.tensor([phase_to_index[name] for name in data["phases"]], dtype=torch.long)
        phase_obs = nn.functional.one_hot(phase_indices, num_classes=len(PHASES)).float()
        obs = torch.cat([obs, phase_obs], dim=-1)

    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    action_mean = actions.mean(dim=0, keepdim=True)
    action_std = actions.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    obs_n = (obs - obs_mean) / obs_std
    actions_n = (actions - action_mean) / action_std

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = PolicyNet(obs.shape[1], actions.shape[1]).to(device)
    loader = DataLoader(TensorDataset(obs_n, actions_n), batch_size=args.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)

    metrics = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_count = 0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred = model(batch_obs)
            loss = nn.functional.mse_loss(pred, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_obs.shape[0]
            total_count += batch_obs.shape[0]
        epoch_loss = total_loss / max(total_count, 1)
        metrics.append({"epoch": epoch, "loss": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(metrics[-1], sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": data["task"],
            "label": data["label"],
            "model_state_dict": model.state_dict(),
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "obs_dim": obs.shape[1],
            "action_dim": actions.shape[1],
            "use_phase": args.use_phase,
            "phases": PHASES,
            "metrics": metrics,
            "dataset": str(Path(args.dataset).resolve()),
        },
        output,
    )
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
