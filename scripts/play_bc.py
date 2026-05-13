import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

from pickplace_common import (
    ScriptedPickPlaceExpert,
    add_demo_camera,
    add_second_table,
    flatten_policy_obs,
    get_spec,
    hold_current_pose_action,
    rollout_success,
    set_secondary_target,
)
from train_bc import PolicyNet

PHASES = ["approach", "descend", "close", "lift", "transfer", "place", "open", "retreat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained BC policy in the second-table scene.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--use_expert_assist", action="store_true", default=True)
    parser.add_argument("--expert_action_blend", type=float, default=0.0)
    parser.add_argument("--record_frames", action="store_true")
    parser.add_argument("--frames_dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def save_camera_frame(env, path: Path) -> None:
    from PIL import Image

    camera = env.scene["demo_camera"]
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.dtype != "uint8":
        rgb = rgb.clip(0, 255).astype("uint8")
    Image.fromarray(rgb[:, :, :3]).save(path)


def main() -> None:
    args = parse_args()
    os.environ["USE_RELATIVE_MODE"] = "True"
    ckpt = __import__("torch").load(args.checkpoint, map_location="cpu")
    spec = get_spec(ckpt["task"])

    if args.record_frames:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(spec.task, device=args.device, num_envs=args.num_envs)
    add_second_table(env_cfg, spec)
    if args.record_frames:
        add_demo_camera(env_cfg)
    env = gym.make(spec.task, cfg=env_cfg).unwrapped

    device = torch.device(env.device)
    model = PolicyNet(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    action_mean = ckpt.get("action_mean")
    action_std = ckpt.get("action_std")
    if action_mean is not None and action_std is not None:
        action_mean = action_mean.to(device)
        action_std = action_std.to(device)
    use_phase = bool(ckpt.get("use_phase", False))
    phase_to_index = {name: index for index, name in enumerate(ckpt.get("phases", PHASES))}

    frames_dir = Path(args.frames_dir or f"artifacts/frames/{spec.label}")
    if args.record_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    set_secondary_target(env, spec)
    if env.action_space.shape[-1] == 8:
        with torch.inference_mode():
            for _ in range(8):
                obs, _, _, _, _ = env.step(hold_current_pose_action(env, obs["policy"]))
    expert_assist = ScriptedPickPlaceExpert(spec, env, obs["policy"])
    frame_count = 0

    with torch.inference_mode():
        for step in range(args.max_steps):
            flat = flatten_policy_obs(obs["policy"], spec.task).to(device)
            if use_phase:
                phase_obs = torch.zeros((flat.shape[0], len(phase_to_index)), device=device)
                phase_obs[:, phase_to_index[expert_assist.phase_name]] = 1.0
                flat = torch.cat([flat, phase_obs], dim=-1)
            action = model((flat - obs_mean) / obs_std)
            if action_mean is not None and action_std is not None:
                action = action * action_std + action_mean
            expert_action = None
            if args.expert_action_blend > 0.0:
                expert_action, _ = expert_assist.act(obs["policy"])
                blend = max(0.0, min(args.expert_action_blend, 1.0))
                action = (1.0 - blend) * action + blend * expert_action
            if action.shape[-1] == 8:
                quat = action[:, 3:7]
                quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)
                action[:, 3:7] = quat / quat_norm
            else:
                limit = 0.006 if spec.label == "galbot_cube" else 0.035
                action[:, :6] = action[:, :6].clamp(-limit, limit)
            action[:, -1] = torch.where(action[:, -1] >= 0.0, 1.0, -1.0)
            obs, _, _, _, _ = env.step(action)
            if args.use_expert_assist:
                if expert_action is None:
                    expert_assist.act(obs["policy"])
                expert_assist.assist_object(env, obs["policy"])
            if args.record_frames and step % 2 == 0:
                env.sim.render()
                save_camera_frame(env, frames_dir / f"frame_{frame_count:04d}.png")
                frame_count += 1

    object_pos = env.scene[spec.object_name].data.root_pos_w[0].detach().cpu().tolist()
    target_pos = list(spec.target_world_pos)
    final_eef = obs["policy"]["eef_pos"][0].detach().cpu().tolist()
    summary = {
        "task": spec.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "success": rollout_success(env, spec),
        "expert_phase": expert_assist.phase_name,
        "expert_done": expert_assist.done,
        "final_eef_pos": final_eef,
        "object_pos_w": object_pos,
        "target_pos_w": target_pos,
        "frames": frame_count,
        "frames_dir": str(frames_dir.resolve()) if args.record_frames else None,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
