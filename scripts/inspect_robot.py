from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect robot articulation metadata for an IsaacLab task.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.scene["robot"]
    joint_names = _as_list(getattr(robot, "joint_names", None))
    body_names = _as_list(getattr(robot, "body_names", None))
    lower_joint_names = [name.lower() for name in joint_names]
    lower_body_names = [name.lower() for name in body_names]

    interesting_tokens = ("wheel", "caster", "base", "drive", "roll", "tire")
    out = {
        "task": args.task,
        "action_space_shape": list(env.action_space.shape),
        "root_pos_w": robot.data.root_pos_w.detach().cpu().reshape(-1).tolist(),
        "root_quat_w": robot.data.root_quat_w.detach().cpu().reshape(-1).tolist(),
        "joint_count": len(joint_names),
        "body_count": len(body_names),
        "joint_names": joint_names,
        "body_names": body_names,
        "interesting_joints": [
            name for name, low in zip(joint_names, lower_joint_names) if any(token in low for token in interesting_tokens)
        ],
        "interesting_bodies": [
            name for name, low in zip(body_names, lower_body_names) if any(token in low for token in interesting_tokens)
        ],
        "policy_obs_keys": sorted(obs["policy"].keys()),
    }

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    os.environ.setdefault("USE_RELATIVE_MODE", "True")
    main()
