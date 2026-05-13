from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug articulation root motion and base-frame observations.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from pickplace_common import base_to_world, get_spec, object_pos_from_obs, set_robot_root_pose

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    spec = get_spec(args.task)
    obs, _ = env.reset()
    robot = env.scene["robot"]

    def snap(label: str, policy_obs):
        eef_b = policy_obs["eef_pos"].reshape(env.num_envs, 3)
        return {
            "label": label,
            "root_pos_w": robot.data.root_pos_w.detach().cpu().reshape(-1).tolist(),
            "eef_pos_b": eef_b.detach().cpu().reshape(-1).tolist(),
            "eef_pos_w_from_root": base_to_world(env, eef_b).detach().cpu().reshape(-1).tolist(),
            "object_pos_b_obs": object_pos_from_obs(policy_obs, spec).detach().cpu().reshape(-1).tolist(),
            "object_pos_w_asset": env.scene[spec.object_name].data.root_pos_w.detach().cpu().reshape(-1).tolist(),
        }

    rows = [snap("reset", obs["policy"])]
    target_root = robot.data.root_pos_w.clone()
    target_root[:, 0] += 3.2
    set_robot_root_pose(env, target_root, robot.data.root_quat_w.clone())
    rows.append(snap("after_write_before_step", obs["policy"]))

    action = torch.zeros((1, env.action_space.shape[-1]), device=env.device)
    action[:, -1] = -1.0
    for index in range(4):
        obs, _, _, _, _ = env.step(action)
        rows.append(snap(f"after_step_{index + 1}", obs["policy"]))

    text = json.dumps(rows, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    os.environ.setdefault("USE_RELATIVE_MODE", "True")
    main()
