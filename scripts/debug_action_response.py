from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--action", nargs="*", type=float, default=None)
    parser.add_argument("--with_second_table", action="store_true")
    parser.add_argument("--output", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from pickplace_common import add_second_table, get_spec, object_pos_from_obs, world_to_base

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    spec = get_spec(args.task)
    if args.with_second_table:
        add_second_table(env_cfg, spec)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    target_w = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    target_b = world_to_base(env, target_w)
    object_b = object_pos_from_obs(obs["policy"], spec)

    action_dim = env.action_space.shape[-1]
    action = torch.zeros((1, action_dim), device=env.device)
    if args.action:
        values = torch.tensor(args.action, dtype=torch.float32, device=env.device)
        action[:, : min(action_dim, len(args.action))] = values[:action_dim]

    rows = []
    for step in range(args.steps):
        policy = obs["policy"]
        rows.append(
            {
                "step": step,
                "eef_pos": policy["eef_pos"].detach().cpu().reshape(-1).tolist(),
                "eef_quat": policy["eef_quat"].detach().cpu().reshape(-1).tolist(),
                "cube_positions": policy.get("cube_positions", torch.empty(1, 0, device=env.device))
                .detach()
                .cpu()
                .reshape(-1)
                .tolist(),
                "mug_positions": policy.get("mug_positions", torch.empty(1, 0, device=env.device))
                .detach()
                .cpu()
                .reshape(-1)
                .tolist(),
                "toy_truck_positions": policy.get("toy_truck_positions", torch.empty(1, 0, device=env.device))
                .detach()
                .cpu()
                .reshape(-1)
                .tolist(),
                "robot_root_pos_w": env.scene["robot"].data.root_pos_w.detach().cpu().reshape(-1).tolist(),
                "robot_root_quat_w": env.scene["robot"].data.root_quat_w.detach().cpu().reshape(-1).tolist(),
                "target_b": target_b.detach().cpu().reshape(-1).tolist(),
                "object_start_b": object_b.detach().cpu().reshape(-1).tolist(),
            }
        )
        obs, _, _, _, _ = env.step(action)

    text = json.dumps(rows, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    os.environ.setdefault("USE_RELATIVE_MODE", "True")
    main()
