import argparse
import contextlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless smoke test for IsaacLab tasks.")
    parser.add_argument("--task", required=True, help="Gymnasium task id.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="artifacts/smoke_result.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def write_json(path: str, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_json(args.output, {"task": args.task, "status": "launching"})
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    write_json(args.output, {"task": args.task, "status": "launched"})

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    os.environ.setdefault("USE_RELATIVE_MODE", "True")

    write_json(args.output, {"task": args.task, "status": "parsing_cfg"})
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.sim.render_interval = max(1, getattr(env_cfg, "decimation", 1))
    write_json(args.output, {"task": args.task, "status": "creating_env"})
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    result: dict[str, object] = {
        "status": "created",
        "task": args.task,
        "num_envs": args.num_envs,
        "device": str(env.device),
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
    }
    write_json(args.output, result)

    result["status"] = "resetting"
    write_json(args.output, result)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        result["observation_keys"] = sorted(obs.keys())
        if "policy" in obs and isinstance(obs["policy"], dict):
            result["policy_keys"] = sorted(obs["policy"].keys())
    result["status"] = "stepping"
    write_json(args.output, result)

    zero_action = torch.zeros((args.num_envs, env.action_space.shape[-1]), device=env.device)
    terminated = torch.zeros((args.num_envs,), dtype=torch.bool, device=env.device)
    truncated = torch.zeros((args.num_envs,), dtype=torch.bool, device=env.device)
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for _ in range(args.steps):
            obs, reward, terminated, truncated, extras = env.step(zero_action)

    result["status"] = "completed"
    result["completed_steps"] = args.steps
    result["terminated_any"] = bool(torch.as_tensor(terminated).any().item())
    result["truncated_any"] = bool(torch.as_tensor(truncated).any().item())
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
