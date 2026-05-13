import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from pickplace_common import (
    ScriptedPickPlaceExpert,
    add_second_table,
    flatten_policy_obs,
    get_spec,
    hold_current_pose_action,
    rollout_success,
    set_secondary_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted pick-move-place demonstrations.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--require_success", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = parse_args()
    os.environ["USE_RELATIVE_MODE"] = "True"
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    spec = get_spec(args.task)
    env_cfg = parse_env_cfg(spec.task, device=args.device, num_envs=args.num_envs)
    add_second_table(env_cfg, spec)
    env = gym.make(spec.task, cfg=env_cfg).unwrapped

    obs_rows = []
    action_rows = []
    phase_rows = []
    successes = 0
    attempts = 0

    with torch.inference_mode():
        max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes
        while attempts < max_attempts and (attempts < args.episodes if not args.require_success else successes < args.episodes):
            attempts += 1
            episode_obs_rows = []
            episode_action_rows = []
            episode_phase_rows = []
            obs, _ = env.reset()
            set_secondary_target(env, spec)
            if env.action_space.shape[-1] == 8:
                for _ in range(8):
                    obs, _, _, _, _ = env.step(hold_current_pose_action(env, obs["policy"]))
            expert = ScriptedPickPlaceExpert(spec, env, obs["policy"])
            for _ in range(args.max_steps):
                policy_obs = obs["policy"]
                episode_obs_rows.append(flatten_policy_obs(policy_obs, spec.task).detach().cpu())
                action, phase = expert.act(policy_obs)
                episode_action_rows.append(action.detach().cpu())
                episode_phase_rows.append(phase)
                obs, _, _, _, _ = env.step(action)
                expert.assist_object(env, obs["policy"])
                if expert.done:
                    break
            success = rollout_success(env, spec)
            successes += int(success)
            if success or not args.require_success:
                obs_rows.extend(episode_obs_rows)
                action_rows.extend(episode_action_rows)
                phase_rows.extend(episode_phase_rows)

    if not obs_rows:
        raise RuntimeError(f"No episodes were collected. successes={successes}, attempts={attempts}")

    dataset = {
        "task": spec.task,
        "label": spec.label,
        "obs": torch.cat(obs_rows, dim=0),
        "actions": torch.cat(action_rows, dim=0),
        "phases": phase_rows,
        "episodes": args.episodes,
        "attempts": attempts,
        "successes": successes,
        "require_success": args.require_success,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)

    summary = {
        "task": spec.task,
        "episodes": args.episodes,
        "attempts": attempts,
        "samples": int(dataset["obs"].shape[0]),
        "successes": successes,
        "success_rate": successes / max(attempts, 1),
        "kept_success_only": args.require_success,
        "output": str(output),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
