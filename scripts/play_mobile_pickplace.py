from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from pickplace_common import (
    MobileTaskSpec,
    add_cube_frame_markers,
    add_demo_camera,
    add_isaac_warehouse_scene,
    add_mobile_demo_camera,
    add_mobile_target_table,
    base_to_world,
    configure_mobile_source_table,
    flatten_policy_obs,
    get_mobile_spec,
    hold_current_pose_action,
    mobile_rollout_success,
    object_pos_from_obs,
    set_mobile_secondary_target,
    set_rigid_pose,
    set_robot_root_pose,
    world_to_base,
)
from train_bc import PolicyNet

PHASES = ["approach", "descend", "close", "lift", "transfer", "place", "open", "retreat"]
BASE_MOTION_PHASES = ("undock", "aisle_drive", "dock")
ROBOT_BASE_RADIUS_BY_LABEL = {
    "galbot_cube": 0.42,
    "agibot_mug": 0.45,
    "agibot_toy2box": 0.45,
}
TABLE_CLEARANCE_THRESHOLD_M = 0.35
CUBE_TABLE_CLEARANCE_THRESHOLD_M = 0.03
GALBOT_CUBE_TCP_PICK_OFFSET_B = (0.540, 0.395)
GALBOT_CUBE_EEF_PICK_NUDGE_B = (0.0, 0.0, 0.0)
GALBOT_CUBE_ATTACHED_OFFSET_B = (0.0, 0.0, -0.035)
GALBOT_CUBE_GRASP_Z = -0.015
AISLE_Y = -1.85
DOCK_Y = -0.95
QHD_WIDTH = 2560
QHD_HEIGHT = 1440


def table_clearance_threshold(spec: MobileTaskSpec) -> float:
    return CUBE_TABLE_CLEARANCE_THRESHOLD_M if spec.base.label == "galbot_cube" else TABLE_CLEARANCE_THRESHOLD_M


def source_object_preferred_y(spec: MobileTaskSpec, source_aabb: dict[str, list[float]]) -> float:
    preferred_y = {
        "galbot_cube": 0.15,
        "agibot_mug": -0.16,
        "agibot_toy2box": -0.18,
    }.get(spec.base.label, 0.0)
    return min(max(preferred_y, source_aabb["min"][1] + 0.10), source_aabb["max"][1] - 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a mobile-base pick/place policy in a far-table scene.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--drive_steps", type=int, default=420)
    parser.add_argument("--expert_action_blend", type=float, default=1.0)
    parser.add_argument("--record_frames", action="store_true")
    parser.add_argument("--frames_dir", default=None)
    parser.add_argument("--capture_backend", choices=("auto", "camera", "viewport"), default="auto")
    parser.add_argument("--use_cfg_camera_pose", action="store_true")
    parser.add_argument("--use_close_demo_camera", action="store_true")
    parser.add_argument("--show_cube_frame_markers", action="store_true")
    parser.add_argument("--skip_industrial_scene", action="store_true")
    parser.add_argument("--skip_source_reposition", action="store_true")
    parser.add_argument("--skip_source_standoff", action="store_true")
    parser.add_argument("--skip_object_reposition", action="store_true")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _rgb_to_uint8(rgb):
    rgb = rgb[:, :, :3]
    if rgb.dtype != "uint8":
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = rgb.clip(0, 255).astype("uint8")
    return rgb


def image_has_content(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 2048:
        return False
    from PIL import Image

    rgb = Image.open(path).convert("RGB")
    extrema = rgb.getextrema()
    return any(high > 4 for _, high in extrema)


def save_camera_frame(env, path: Path, dt: float) -> bool:
    from PIL import Image

    camera = env.scene["demo_camera"]
    camera.update(dt, force_recompute=True)
    output = camera.data.output
    rgb = output["rgb"][0].detach().cpu().numpy()
    if rgb.max() <= 0 and "rgba" in output:
        rgb = output["rgba"][0].detach().cpu().numpy()
    rgb = _rgb_to_uint8(rgb)
    Image.fromarray(rgb[:, :, :3]).save(path)
    return bool(rgb.max() > 4)


def quaternion_angle_error(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-6)
    q_ref = q_ref / torch.linalg.vector_norm(q_ref, dim=-1, keepdim=True).clamp_min(1.0e-6)
    dot = torch.abs(torch.sum(q * q_ref, dim=-1)).clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def set_usd_world_pose(stage, prim_path: str, pos: torch.Tensor, quat: torch.Tensor) -> bool:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
        xform.ClearXformOpOrder()
        op = xform.AddTransformOp()
    else:
        op = ops[0]
    matrix = Gf.Matrix4d(1.0)
    rotation = Gf.Rotation(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
    matrix.SetRotate(rotation)
    matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    op.Set(matrix)
    return True


def update_cube_frame_markers(env, cube_pos_w: torch.Tensor, cube_quat_w: torch.Tensor) -> bool:
    import isaaclab.utils.math as math_utils
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env_candidates = ["/World/envs/env_0", str(getattr(env.scene, "env_ns", "/World/envs/env_0"))]
    env_path = env_candidates[0]
    for candidate in env_candidates:
        if stage.GetPrimAtPath(f"{candidate}/CubeFrame_X").IsValid():
            env_path = candidate
            break
    marker_paths = [
        f"{env_path}/CubeFrame_X",
        f"{env_path}/CubeFrame_Y",
        f"{env_path}/CubeFrame_Z",
    ]
    axis_len = 0.18
    lift = 0.058
    local_offsets = torch.tensor(
        [
            [axis_len * 0.5, 0.0, lift],
            [0.0, axis_len * 0.5, lift],
            [0.0, 0.0, lift + axis_len * 0.5],
        ],
        dtype=torch.float32,
        device=env.device,
    )
    local_quats = torch.zeros((3, 4), dtype=torch.float32, device=env.device)
    local_quats[:, 0] = 1.0
    pos_w, quat_w = math_utils.combine_frame_transforms(
        cube_pos_w[:1].repeat(3, 1),
        cube_quat_w[:1].repeat(3, 1),
        local_offsets,
        local_quats,
    )
    ok = True
    for index, marker_path in enumerate(marker_paths):
        ok = set_usd_world_pose(stage, marker_path, pos_w[index], quat_w[index]) and ok
    return ok


def configure_stable_galbot_gripper(env_cfg, spec: MobileTaskSpec) -> None:
    if spec.base.label != "galbot_cube":
        return
    gripper_action = getattr(getattr(env_cfg, "actions", None), "gripper_action", None)
    if gripper_action is None:
        return
    if hasattr(gripper_action, "close_command_expr"):
        gripper_action.close_command_expr = {"left_gripper_.*_joint": 0.018}
    if hasattr(gripper_action, "open_command_expr"):
        gripper_action.open_command_expr = {"left_gripper_.*_joint": 0.035}


def get_demo_camera_path(env) -> str:
    camera = env.scene["demo_camera"]
    sensor_prims = getattr(camera, "_sensor_prims", None)
    if sensor_prims:
        return str(sensor_prims[0].GetPath())
    return "/World/envs/env_0/DemoCamera"


class ViewportFrameCapture:
    def __init__(self, camera_path: str, width: int = QHD_WIDTH, height: int = QHD_HEIGHT):
        self.camera_path = camera_path
        self.width = width
        self.height = height
        self.viewport = None
        self.window = None

    def ensure(self) -> bool:
        from pxr import Sdf
        import omni.kit.viewport.utility as vp_utils

        if self.viewport is None:
            self.viewport = vp_utils.get_active_viewport()
        if self.viewport is None:
            self.window = vp_utils.create_viewport_window(
                "DemoCaptureViewport",
                width=self.width,
                height=self.height,
                camera_path=Sdf.Path(self.camera_path),
            )
            if self.window is not None:
                self.viewport = self.window.viewport_api
        if self.viewport is None:
            return False
        self.viewport.camera_path = Sdf.Path(self.camera_path)
        self.viewport.resolution = (self.width, self.height)
        return True

    def save_frame(self, env, path: Path) -> bool:
        import omni.kit.viewport.utility as vp_utils

        if not self.ensure():
            return False
        if path.exists():
            path.unlink()
        for _ in range(2):
            env.sim.render()
        vp_utils.capture_viewport_to_file(self.viewport, file_path=str(path))
        for _ in range(30):
            env.sim.render()
            if image_has_content(path):
                return True
        return image_has_content(path)


def get_table_aabbs_xy() -> dict[str, dict[str, list[float]]]:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out: dict[str, dict[str, list[float]]] = {}
    for name, prim_path in (("source", "/World/envs/env_0/Table"), ("target", "/World/envs/env_0/TargetTable")):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = bbox.GetMin()
        mx = bbox.GetMax()
        out[name] = {"min": [float(mn[0]), float(mn[1])], "max": [float(mx[0]), float(mx[1])]}
    return out


def point_to_aabb_clearance(point_xy: torch.Tensor, aabb: dict[str, list[float]], robot_radius: float) -> torch.Tensor:
    mn = torch.tensor(aabb["min"], dtype=torch.float32, device=point_xy.device)
    mx = torch.tensor(aabb["max"], dtype=torch.float32, device=point_xy.device)
    outside = torch.maximum(torch.maximum(mn - point_xy, point_xy - mx), torch.zeros_like(point_xy))
    outside_dist = torch.linalg.vector_norm(outside, dim=-1)
    inside = ((point_xy >= mn) & (point_xy <= mx)).all(dim=-1)
    inside_margin = torch.stack(
        [point_xy[..., 0] - mn[0], mx[0] - point_xy[..., 0], point_xy[..., 1] - mn[1], mx[1] - point_xy[..., 1]],
        dim=-1,
    ).min(dim=-1).values
    signed_distance = torch.where(inside, -inside_margin, outside_dist)
    return signed_distance - robot_radius


def route_clearance(
    route_w: torch.Tensor,
    table_aabbs_xy: dict[str, dict[str, list[float]]],
    robot_radius: float,
    samples_per_segment: int = 31,
) -> tuple[float, torch.Tensor]:
    samples = []
    for segment_index in range(route_w.shape[0] - 1):
        start = route_w[segment_index, :2]
        end = route_w[segment_index + 1, :2]
        for sample_index in range(samples_per_segment):
            if segment_index > 0 and sample_index == 0:
                continue
            raw = float(sample_index) / float(samples_per_segment - 1)
            t = raw * raw * (3.0 - 2.0 * raw)
            samples.append((1.0 - t) * start + t * end)
    sample_tensor = torch.stack(samples, dim=0)
    clearances = []
    for aabb in table_aabbs_xy.values():
        clearances.append(point_to_aabb_clearance(sample_tensor, aabb, robot_radius))
    if not clearances:
        return float("inf"), sample_tensor
    all_clearances = torch.stack(clearances, dim=-1)
    return float(all_clearances.min().item()), sample_tensor


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def smooth_step(raw: torch.Tensor | float) -> torch.Tensor | float:
    return raw * raw * (3.0 - 2.0 * raw)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    half_yaw = 0.5 * yaw
    quat[:, 0] = torch.cos(half_yaw)
    quat[:, 3] = torch.sin(half_yaw)
    return quat


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lerp_yaw(start_yaw: torch.Tensor, end_yaw: torch.Tensor, t: float) -> torch.Tensor:
    return start_yaw + wrap_angle(end_yaw - start_yaw) * t


def target_in_base_from_plan_pose(target_w: torch.Tensor, root_pos_w: torch.Tensor, root_yaw: torch.Tensor) -> torch.Tensor:
    delta = target_w - root_pos_w
    cos_yaw = torch.cos(root_yaw)
    sin_yaw = torch.sin(root_yaw)
    target_b = delta.clone()
    target_b[:, 0] = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    target_b[:, 1] = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return target_b


def move_to_safe_source_standoff(
    env,
    obs: dict,
    spec: MobileTaskSpec,
    table_aabbs_xy: dict[str, dict[str, list[float]]],
) -> tuple[dict, list[float] | None]:
    source_aabb = table_aabbs_xy.get("source")
    if source_aabb is None:
        return obs, None
    robot_radius = ROBOT_BASE_RADIUS_BY_LABEL.get(spec.base.label, 0.45)
    if spec.base.label == "galbot_cube":
        object_x = source_aabb["min"][0] + 0.08
        object_y = source_object_preferred_y(spec, source_aabb)
        safe_x = object_x - GALBOT_CUBE_TCP_PICK_OFFSET_B[0]
        safe_y = object_y - GALBOT_CUBE_TCP_PICK_OFFSET_B[1]
    else:
        safe_x = source_aabb["min"][0] - robot_radius - table_clearance_threshold(spec) - 0.04
        safe_y = None
    robot = env.scene["robot"]
    safe_pos = robot.data.root_pos_w.clone()
    already_safe_x = float(safe_pos[0, 0].item()) <= safe_x
    already_safe_y = safe_y is None or abs(float(safe_pos[0, 1].item()) - safe_y) < 1.0e-4
    if already_safe_x and already_safe_y:
        return obs, safe_pos[0].detach().cpu().tolist()
    safe_pos[:, 0] = safe_x
    if safe_y is not None:
        safe_pos[:, 1] = safe_y
    set_robot_root_pose(env, safe_pos, robot.data.root_quat_w.clone())
    with torch.inference_mode():
        for _ in range(6):
            obs, _, _, _, _ = env.step(hold_current_pose_action(env, obs["policy"], gripper=1.0))
    return obs, safe_pos[0].detach().cpu().tolist()


def place_source_object_near_safe_edge(
    env,
    obs: dict,
    spec: MobileTaskSpec,
    table_aabbs_xy: dict[str, dict[str, list[float]]],
) -> tuple[dict, list[float] | None]:
    source_aabb = table_aabbs_xy.get("source")
    if source_aabb is None:
        return obs, None
    asset = env.scene[spec.base.object_name]
    object_pos = asset.data.root_pos_w.clone()
    object_quat = asset.data.root_quat_w.clone()
    edge_margin_x = 0.08 if spec.base.label != "agibot_mug" else 0.12
    object_pos[:, 0] = source_aabb["min"][0] + edge_margin_x
    object_pos[:, 1] = source_object_preferred_y(spec, source_aabb)
    if spec.base.label == "galbot_cube":
        object_quat.zero_()
        object_quat[:, 0] = 1.0
        object_pos[:, 2] = spec.source_table_pos[2] + spec.target_world_pos[2]
    set_rigid_pose(env, spec.base.object_name, object_pos, object_quat)
    return obs, object_pos[0].detach().cpu().tolist()


def hide_unused_galbot_cubes(env, spec: MobileTaskSpec) -> list[str]:
    import omni.usd
    from pxr import Usd, UsdGeom

    if spec.base.label != "galbot_cube":
        return []
    stage = omni.usd.get_context().get_stage()
    hidden_names: list[str] = []
    for index, name in enumerate(("cube_1", "cube_2")):
        try:
            asset = env.scene[name]
        except KeyError:
            continue
        hidden_pos = torch.tensor([[-20.0 - float(index), 12.0, -8.0]], dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        hidden_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
        hidden_quat[:, 0] = 1.0
        set_rigid_pose(env, name, hidden_pos, hidden_quat)
        try:
            asset.set_visibility(False)
        except AttributeError:
            pass
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Cube_{index + 1}")
        if prim.IsValid():
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(child).MakeInvisible()
        hidden_names.append(name)
    return hidden_names


class MobilePickPlaceController:
    phase_names = [
        "approach",
        "descend",
        "close",
        "lift",
        "undock",
        "aisle_drive",
        "dock",
        "transfer",
        "place",
        "open",
        "retreat",
    ]

    def __init__(
        self,
        spec: MobileTaskSpec,
        env,
        initial_policy_obs: dict[str, torch.Tensor],
        drive_steps: int,
        table_aabbs_xy: dict[str, dict[str, list[float]]] | None = None,
        root_start_override_w: list[float] | None = None,
    ):
        self.spec = spec
        self.base = spec.base
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.action_dim = env.action_space.shape[-1]
        self.drive_steps = max(1, drive_steps)
        self.phase_index = 0
        self.hold_count = 0
        self.drive_count = 0
        self.motion_phase_count = 0
        self.phase_step_count = 0
        self.attached = False
        self.released = False
        self.max_object_step_m = 0.0
        self.release_settle_steps = 0
        self.base_motion_phases = BASE_MOTION_PHASES
        self.robot_base_radius = ROBOT_BASE_RADIUS_BY_LABEL.get(self.base.label, 0.45)
        self.clearance_threshold_m = table_clearance_threshold(spec)
        self.table_aabbs_xy = table_aabbs_xy or {}
        robot = env.scene["robot"]
        if root_start_override_w is None:
            self.root_start_pos = robot.data.root_pos_w.clone()
        else:
            self.root_start_pos = torch.tensor(root_start_override_w, dtype=torch.float32, device=env.device).repeat(
                env.num_envs, 1
            )
        self.root_start_quat = robot.data.root_quat_w.clone()
        self.root_start_yaw = quat_to_yaw(self.root_start_quat)
        set_robot_root_pose(env, self.root_start_pos, self.root_start_quat)

        object_asset = env.scene[self.base.object_name]
        self.source_object_pos_w = object_asset.data.root_pos_w.clone()
        self.source_object_quat_w = object_asset.data.root_quat_w.clone()
        if self.base.label == "galbot_cube":
            self.source_object_quat_w.zero_()
            self.source_object_quat_w[:, 0] = 1.0
        self.object_cmd_pos_w = self.source_object_pos_w.clone()
        self.object_cmd_quat_w = self.source_object_quat_w.clone()
        self.object_locked_quat_w = self.source_object_quat_w.clone()
        if self.base.label == "galbot_cube":
            set_rigid_pose(env, self.base.object_name, self.source_object_pos_w, self.object_locked_quat_w)
        self.object_start_b = world_to_base(env, self.source_object_pos_w)
        self.current_eef_target_b = self.object_start_b + self._z(self.base.approach_z)

        near_target_w = torch.tensor(self.base.target_world_pos, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        far_target_w = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        static_target_b = target_in_base_from_plan_pose(near_target_w, self.root_start_pos, self.root_start_yaw)
        nominal_goal_pos = far_target_w - static_target_b
        self.root_goal_pos, self.root_goal_yaw = self._select_docking_pose(
            nominal_goal_pos, far_target_w, static_target_b
        )
        self.root_goal_quat = yaw_to_quat(self.root_goal_yaw)
        aisle_y = torch.full_like(self.root_start_pos[:, 1:2], AISLE_Y)
        self.route_waypoints = torch.stack(
            [
                self.root_start_pos,
                torch.cat([self.root_start_pos[:, 0:1], aisle_y, self.root_start_pos[:, 2:3]], dim=-1),
                torch.cat([self.root_goal_pos[:, 0:1], aisle_y, self.root_goal_pos[:, 2:3]], dim=-1),
                self.root_goal_pos,
            ],
            dim=1,
        )
        self.segment_steps = self._split_drive_steps(self.drive_steps)
        self.min_table_clearance_m, self.route_samples_xy = route_clearance(
            self.route_waypoints[0], self.table_aabbs_xy, self.robot_base_radius
        )
        self.collision_check_passed = self.min_table_clearance_m >= self.clearance_threshold_m
        self.segment_drive_yaws = self._route_segment_yaws(self.route_waypoints)
        self.segment_start_yaws = torch.stack(
            [self.root_start_yaw, self.segment_drive_yaws[:, 0], self.segment_drive_yaws[:, 1]], dim=1
        )
        self.segment_end_yaws = torch.stack(
            [self.segment_drive_yaws[:, 0], self.segment_drive_yaws[:, 1], self.root_goal_yaw], dim=1
        )

    def _select_docking_pose(
        self, nominal_goal_pos: torch.Tensor, far_target_w: torch.Tensor, static_target_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_aabb = self.table_aabbs_xy.get("target")
        front_gap = self.robot_base_radius + self.clearance_threshold_m + 0.03
        if target_aabb is not None:
            front_y = target_aabb["min"][1] - front_gap
            y_candidates = [front_y, front_y - 0.08, front_y - 0.16, front_y - 0.24]
        else:
            y_candidates = [-1.60, -1.68, -1.76, -1.84]
        x_offsets = [0.0, -0.12, 0.12, -0.24, 0.24, -0.36, 0.36, -0.48, 0.48]
        best = nominal_goal_pos.clone()
        best_yaw = self.root_start_yaw.clone()
        best_clearance = -1.0e9
        best_score = -1.0e9
        for y_value in y_candidates:
            for x_offset in x_offsets:
                candidate = self.root_start_pos.clone()
                candidate[:, 0] = far_target_w[:, 0] + x_offset
                candidate[:, 1] = y_value
                candidate[:, 2] = self.root_start_pos[:, 2]
                target_delta = far_target_w[:, :2] - candidate[:, :2]
                candidate_yaw = torch.atan2(target_delta[:, 1], target_delta[:, 0])
                target_b = target_in_base_from_plan_pose(far_target_w, candidate, candidate_yaw)
                if not self._target_within_reach_window(target_b, static_target_b):
                    continue
                route = torch.stack(
                    [
                        self.root_start_pos[0],
                        torch.tensor(
                            [self.root_start_pos[0, 0].item(), AISLE_Y, self.root_start_pos[0, 2].item()],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        torch.tensor(
                            [candidate[0, 0].item(), AISLE_Y, candidate[0, 2].item()],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        candidate[0],
                    ],
                    dim=0,
                )
                clearance, _ = route_clearance(route, self.table_aabbs_xy, self.robot_base_radius)
                target_distance = float(torch.linalg.vector_norm(target_b[:, :2], dim=-1).max().item())
                lateral_error = float(torch.abs(target_b[:, 1]).max().item())
                score = -target_distance - 0.15 * lateral_error + min(clearance, 1.0) * 0.10
                if clearance > best_clearance:
                    best_clearance = clearance
                if score > best_score:
                    best_score = score
                    best = candidate.clone()
                    best_yaw = candidate_yaw.clone()
                if clearance >= self.clearance_threshold_m:
                    return candidate, candidate_yaw
        return best, best_yaw

    def _target_within_reach_window(self, target_b: torch.Tensor, static_target_b: torch.Tensor) -> bool:
        if self.base.label == "galbot_cube":
            x_min, x_max, y_abs_max = 0.45, 1.36, 0.60
        else:
            x_min, x_max, y_abs_max = 0.45, 1.36, 0.60
        x_ok = bool(((target_b[:, 0] > x_min) & (target_b[:, 0] < x_max)).all().item())
        y_ok = bool((torch.abs(target_b[:, 1]) < y_abs_max).all().item())
        z_ok = bool((torch.abs(target_b[:, 2] - static_target_b[:, 2]) < 0.25).all().item())
        return x_ok and y_ok and z_ok

    @staticmethod
    def _route_segment_yaws(route_waypoints: torch.Tensor) -> torch.Tensor:
        deltas = route_waypoints[:, 1:, :2] - route_waypoints[:, :-1, :2]
        return torch.atan2(deltas[:, :, 1], deltas[:, :, 0])

    @staticmethod
    def _split_drive_steps(drive_steps: int) -> list[int]:
        undock = max(12, int(round(float(drive_steps) * 0.22)))
        dock = max(12, int(round(float(drive_steps) * 0.22)))
        aisle = max(20, drive_steps - undock - dock)
        return [undock, aisle, dock]

    def _advance_phase(self) -> None:
        self.phase_index += 1
        self.phase_step_count = 0
        self.motion_phase_count = 0

    def _z(self, value: float) -> torch.Tensor:
        out = torch.zeros((self.num_envs, 3), device=self.device)
        out[:, 2] = value
        return out

    def _target_b(self) -> torch.Tensor:
        target_w = torch.tensor(self.spec.target_world_pos, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        return world_to_base(self.env, target_w)

    def _cube_pick_target_b(self, z_offset: float) -> torch.Tensor:
        target = self.object_start_b + self._z(z_offset)
        if self.base.label == "galbot_cube":
            nudge = torch.tensor(
                GALBOT_CUBE_EEF_PICK_NUDGE_B, dtype=torch.float32, device=self.device
            ).repeat(self.num_envs, 1)
            target = target + nudge
        return target

    def _phase_target(self) -> tuple[torch.Tensor, float, int, bool, bool]:
        open_cmd = 1.0
        close_cmd = -1.0
        phase = self.phase_name
        if phase == "approach":
            return self._cube_pick_target_b(self.base.approach_z), open_cmd, 0, False, False
        if phase == "descend":
            grasp_z = GALBOT_CUBE_GRASP_Z if self.base.label == "galbot_cube" else self.base.grasp_z
            return self._cube_pick_target_b(grasp_z), open_cmd, 0, False, False
        if phase == "close":
            close_hold_steps = 30 if self.base.label == "galbot_cube" else 18
            grasp_z = GALBOT_CUBE_GRASP_Z if self.base.label == "galbot_cube" else self.base.grasp_z
            return self._cube_pick_target_b(grasp_z), close_cmd, close_hold_steps, True, False
        if phase == "lift":
            return self._cube_pick_target_b(self.base.lift_z), close_cmd, 0, True, False
        target_b = self._target_b()
        if phase == "transfer":
            return target_b + self._z(self.base.lift_z), close_cmd, 0, True, False
        if phase == "place":
            return target_b + self._z(self.base.place_z), close_cmd, 0, True, False
        if phase == "open":
            open_hold_steps = 24 if self.base.label == "galbot_cube" else 18
            return target_b + self._z(self.base.place_z), open_cmd, open_hold_steps, False, True
        return target_b + self._z(self.base.lift_z), open_cmd, 0, False, True

    def act(self, policy_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
        if self.phase_name in self.base_motion_phases:
            return hold_current_pose_action(self.env, policy_obs, gripper=-1.0), self.phase_name

        target, gripper, hold_steps, attach, release = self._phase_target()
        self.current_eef_target_b = target.clone()
        eef = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        delta = target - eef
        dist = torch.linalg.vector_norm(delta, dim=-1)

        action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        if self.action_dim == 8:
            action[:, :3] = base_to_world(self.env, target)
            action[:, 3:7] = policy_obs["eef_quat"].reshape(self.num_envs, 4)
            action[:, 7] = gripper
        else:
            max_delta = 0.006 if self.base.label == "galbot_cube" else 0.035
            scale = torch.clamp(max_delta / (dist + 1.0e-6), max=1.0).unsqueeze(-1)
            action[:, :3] = delta * scale
            action[:, 6] = gripper

        if release:
            self.attached = False
            self.released = True
        elif attach and hold_steps == 0:
            self.attached = True

        if hold_steps > 0:
            self.hold_count += 1
            if self.hold_count >= hold_steps:
                if attach:
                    self.attached = True
                self._advance_phase()
                self.hold_count = 0
        else:
            tolerance = 0.025
            if self.base.label == "galbot_cube":
                if self.phase_name in ("approach", "descend", "close"):
                    tolerance = 0.025
                elif self.phase_name in ("lift", "transfer", "place", "retreat"):
                    tolerance = 0.045
                else:
                    tolerance = 0.035
            elif self.phase_name in ("transfer", "place", "retreat"):
                tolerance = 0.04
            if bool((dist < tolerance).all().item()):
                self._advance_phase()
                self.hold_count = 0

        return action, self.phase_names[min(self.phase_index, len(self.phase_names) - 1)]

    def after_step(self) -> None:
        if self.phase_index > self.phase_names.index("dock"):
            set_robot_root_pose(self.env, self.root_goal_pos, self.root_goal_quat)
            return
        if self.phase_index < self.phase_names.index("undock"):
            set_robot_root_pose(self.env, self.root_start_pos, self.root_start_quat)
            return
        if self.phase_name not in self.base_motion_phases:
            return
        phase_index = self.base_motion_phases.index(self.phase_name)
        segment_steps = self.segment_steps[phase_index]
        self.drive_count += 1
        self.motion_phase_count += 1
        pos, quat = self._nonholonomic_pose_for_segment(phase_index, self.motion_phase_count, segment_steps)
        set_robot_root_pose(self.env, pos, quat)
        if self.motion_phase_count >= segment_steps:
            self._advance_phase()

    def _nonholonomic_pose_for_segment(
        self, phase_index: int, elapsed_steps: int, segment_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.route_waypoints[:, phase_index, :]
        end = self.route_waypoints[:, phase_index + 1, :]
        start_yaw = self.segment_start_yaws[:, phase_index]
        drive_yaw = self.segment_drive_yaws[:, phase_index]
        end_yaw = self.segment_end_yaws[:, phase_index]
        turn_in_steps = min(max(14, int(round(float(segment_steps) * 0.24))), max(segment_steps - 2, 1))
        turn_out_steps = 0
        if phase_index == len(self.base_motion_phases) - 1:
            final_turn = torch.abs(wrap_angle(end_yaw - drive_yaw)).max().item()
            if final_turn > 0.03:
                turn_out_steps = min(max(14, int(round(float(segment_steps) * 0.18))), max(segment_steps - turn_in_steps - 1, 0))
        drive_steps = max(segment_steps - turn_in_steps - turn_out_steps, 1)

        if elapsed_steps <= turn_in_steps:
            raw = min(float(elapsed_steps) / float(turn_in_steps), 1.0)
            yaw = lerp_yaw(start_yaw, drive_yaw, smooth_step(raw))
            return start, yaw_to_quat(yaw)

        drive_elapsed = elapsed_steps - turn_in_steps
        if drive_elapsed <= drive_steps:
            raw = min(float(drive_elapsed) / float(drive_steps), 1.0)
            t = smooth_step(raw)
            pos = (1.0 - t) * start + t * end
            return pos, yaw_to_quat(drive_yaw)

        raw = min(float(elapsed_steps - turn_in_steps - drive_steps) / float(max(turn_out_steps, 1)), 1.0)
        yaw = lerp_yaw(drive_yaw, end_yaw, smooth_step(raw))
        return end, yaw_to_quat(yaw)

    def tick_phase(self) -> None:
        self.phase_step_count += 1
        max_steps = {"approach": 130, "descend": 150, "lift": 140, "transfer": 190, "place": 140, "retreat": 90}
        if self.phase_name in max_steps and self.phase_step_count >= max_steps[self.phase_name]:
            self._advance_phase()

    def assist_object(self, env, policy_obs: dict[str, torch.Tensor]) -> None:
        if self.base.label == "galbot_cube":
            quat = self.object_locked_quat_w.clone()
        else:
            quat = torch.tensor(self.base.carry_quat_world, dtype=torch.float32, device=env.device).repeat(
                env.num_envs, 1
            )
        if self.attached:
            offset = torch.tensor(self.base.hold_offset, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
            if self.base.label == "galbot_cube":
                offset = torch.tensor(
                    GALBOT_CUBE_ATTACHED_OFFSET_B, dtype=torch.float32, device=env.device
                ).repeat(env.num_envs, 1)
                object_pos_b = self.current_eef_target_b + offset
                if self.phase_name == "lift":
                    max_step = 0.010
                elif self.phase_name in self.base_motion_phases:
                    max_step = 0.070
                elif self.phase_name in ("transfer", "place"):
                    max_step = 0.045
                else:
                    max_step = 0.025
            else:
                object_pos_b = policy_obs["eef_pos"].reshape(env.num_envs, 3) + offset
                max_step = 0.050 if self.phase_name in self.base_motion_phases else 0.035
            self._move_object_toward(env, base_to_world(env, object_pos_b), quat, max_step)
        elif not self.released and self.phase_name in ("approach", "descend", "close"):
            self.object_cmd_pos_w = self.source_object_pos_w.clone()
            self.object_cmd_quat_w = self.source_object_quat_w.clone()
            set_rigid_pose(env, self.base.object_name, self.source_object_pos_w, self.source_object_quat_w)
        elif self.released:
            target_w = torch.tensor(self.spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(
                env.num_envs, 1
            )
            self.release_settle_steps += 1
            self._move_object_toward(env, target_w, quat, 0.030)

    def _move_object_toward(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        max_step_m: float,
    ) -> None:
        current_pos = self.object_cmd_pos_w.clone()
        delta = target_pos_w - current_pos
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        ratio = torch.clamp(max_step_m / (dist + 1.0e-6), max=1.0)
        new_pos = current_pos + delta * ratio
        step = torch.linalg.vector_norm(new_pos - current_pos, dim=-1).max().item()
        self.max_object_step_m = max(self.max_object_step_m, float(step))
        self.object_cmd_pos_w = new_pos.clone()
        self.object_cmd_quat_w = target_quat_w.clone()
        set_rigid_pose(env, self.base.object_name, new_pos, target_quat_w)

    @property
    def phase_name(self) -> str:
        return self.phase_names[min(self.phase_index, len(self.phase_names) - 1)]

    @property
    def done(self) -> bool:
        return self.phase_index >= len(self.phase_names)


def load_policy(checkpoint: str | None, device: torch.device):
    if checkpoint is None:
        return None
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = PolicyNet(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "checkpoint": checkpoint,
        "ckpt": ckpt,
        "model": model,
        "obs_mean": ckpt["obs_mean"].to(device),
        "obs_std": ckpt["obs_std"].to(device),
        "action_mean": ckpt.get("action_mean").to(device) if ckpt.get("action_mean") is not None else None,
        "action_std": ckpt.get("action_std").to(device) if ckpt.get("action_std") is not None else None,
        "use_phase": bool(ckpt.get("use_phase", False)),
        "phase_to_index": {name: index for index, name in enumerate(ckpt.get("phases", PHASES))},
    }


def policy_action(policy, obs, spec: MobileTaskSpec, phase_name: str, env) -> torch.Tensor:
    flat = flatten_policy_obs(obs["policy"], spec.base.task).to(env.device)
    if policy["use_phase"]:
        phase_to_index = policy["phase_to_index"]
        phase_obs = torch.zeros((flat.shape[0], len(phase_to_index)), device=env.device)
        if phase_name in phase_to_index:
            phase_obs[:, phase_to_index[phase_name]] = 1.0
        flat = torch.cat([flat, phase_obs], dim=-1)
    action = policy["model"]((flat - policy["obs_mean"]) / policy["obs_std"])
    if policy["action_mean"] is not None and policy["action_std"] is not None:
        action = action * policy["action_std"] + policy["action_mean"]
    if action.shape[-1] == 8:
        quat = action[:, 3:7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-6)
        action[:, 3:7] = quat / quat_norm
    else:
        limit = 0.006 if spec.base.label == "galbot_cube" else 0.035
        action[:, :6] = action[:, :6].clamp(-limit, limit)
    action[:, -1] = torch.where(action[:, -1] >= 0.0, 1.0, -1.0)
    return action


def main() -> None:
    args = parse_args()
    os.environ["USE_RELATIVE_MODE"] = "True"

    if args.checkpoint is None and args.task is None:
        raise ValueError("Provide --checkpoint or --task.")

    ckpt_task = None
    if args.checkpoint is not None:
        ckpt_task = torch.load(args.checkpoint, map_location="cpu")["task"]
    task = args.task or ckpt_task
    spec = get_mobile_spec(task)

    if args.record_frames:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(spec.base.task, device=args.device, num_envs=args.num_envs)
    env_step_s = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args.max_steps) * env_step_s * 1.2)
    configure_stable_galbot_gripper(env_cfg, spec)
    configure_mobile_source_table(env_cfg, spec)
    add_mobile_target_table(env_cfg, spec)
    if not args.skip_industrial_scene:
        add_isaac_warehouse_scene(env_cfg, spec)
    if spec.base.label == "galbot_cube" and args.show_cube_frame_markers:
        add_cube_frame_markers(env_cfg)
    if args.record_frames:
        if args.use_close_demo_camera:
            add_demo_camera(env_cfg)
        else:
            add_mobile_demo_camera(env_cfg)
    env = gym.make(spec.base.task, cfg=env_cfg).unwrapped

    device = torch.device(env.device)
    policy = load_policy(args.checkpoint, device)

    frames_dir = Path(args.frames_dir or f"artifacts/frames/{spec.base.label}_mobile")
    if args.record_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    hidden_cube_names = hide_unused_galbot_cubes(env, spec)
    table_aabbs_xy = get_table_aabbs_xy()
    set_mobile_secondary_target(env, spec)
    if args.record_frames and not args.use_cfg_camera_pose:
        if spec.base.label == "galbot_cube":
            eye_cfg = [2.55, -7.15, 3.05]
            lookat_cfg = [2.55, -0.55, 0.12]
        else:
            eye_cfg = [2.55, -6.20, 2.75]
            lookat_cfg = [2.55, -0.25, 0.05]
        eye = torch.tensor([eye_cfg], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        lookat = torch.tensor([lookat_cfg], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        env.scene["demo_camera"].set_world_poses_from_view(eye, lookat)
        for _ in range(4):
            env.sim.render()
            env.scene["demo_camera"].update(env_step_s, force_recompute=True)
    elif args.record_frames:
        for _ in range(4):
            env.sim.render()
            env.scene["demo_camera"].update(env_step_s, force_recompute=True)
    viewport_capture = None
    if args.record_frames and args.capture_backend in ("auto", "viewport"):
        viewport_capture = ViewportFrameCapture(get_demo_camera_path(env))
        if not viewport_capture.ensure():
            print("WARNING: viewport capture backend is unavailable; falling back to Camera sensor only.", flush=True)
            viewport_capture = None
    if env.action_space.shape[-1] == 8:
        with torch.inference_mode():
            for _ in range(8):
                obs, _, _, _, _ = env.step(hold_current_pose_action(env, obs["policy"]))

    if args.skip_source_reposition:
        source_standoff_pose = None
        source_object_pose = None
    else:
        if args.skip_source_standoff:
            source_standoff_pose = None
        else:
            obs, source_standoff_pose = move_to_safe_source_standoff(env, obs, spec, table_aabbs_xy)
        if args.skip_object_reposition:
            source_object_pose = None
        else:
            obs, source_object_pose = place_source_object_near_safe_edge(env, obs, spec, table_aabbs_xy)
    controller = MobilePickPlaceController(
        spec,
        env,
        obs["policy"],
        args.drive_steps,
        table_aabbs_xy,
        root_start_override_w=source_standoff_pose,
    )
    table_delta = torch.tensor(spec.target_table_pos[:2]) - torch.tensor(spec.source_table_pos[:2])
    table_distance = float(torch.linalg.vector_norm(table_delta).item())
    base_direct_displacement = float(
        torch.linalg.vector_norm(controller.root_goal_pos[0, :2] - controller.root_start_pos[0, :2]).item()
    )
    curve_samples = controller.route_samples_xy
    curve_deltas = curve_samples[1:] - curve_samples[:-1]
    base_route_length = float(torch.linalg.vector_norm(curve_deltas, dim=-1).sum().item())
    table_centers = torch.tensor(
        [spec.source_table_pos[:2], spec.target_table_pos[:2]], dtype=torch.float32, device=curve_samples.device
    )
    drive_distances = torch.linalg.vector_norm(curve_samples[:, None, :] - table_centers[None, :, :], dim=-1)
    drive_min_table_center_clearance = float(drive_distances.min().item())

    frame_count = 0
    done_step = None
    command_success = False
    target_w_for_success = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(
        env.num_envs, 1
    )
    phase_log: list[dict[str, object]] = []
    last_phase = None
    object_quat_ref_w = controller.object_locked_quat_w.clone()
    object_quat_max_angle_error_rad = 0.0
    gripper_command_events: list[dict[str, object]] = []
    last_gripper_sign = None
    gripper_command_transitions = 0
    gripper_pos_min = None
    gripper_pos_max = None
    gripper_pos_prev = None
    gripper_pos_max_step = 0.0
    cube_frame_markers_enabled = False
    if spec.base.label == "galbot_cube" and args.show_cube_frame_markers:
        cube_frame_markers_enabled = update_cube_frame_markers(
            env,
            controller.object_cmd_pos_w,
            controller.object_cmd_quat_w,
        )
    with torch.inference_mode():
        for step in range(args.max_steps):
            phase = controller.phase_name
            expert_action, _ = controller.act(obs["policy"])
            if policy is not None and phase not in controller.base_motion_phases:
                learned_action = policy_action(policy, obs, spec, phase if phase in PHASES else "transfer", env)
                blend = max(0.0, min(args.expert_action_blend, 1.0))
                action = (1.0 - blend) * learned_action + blend * expert_action
            else:
                action = expert_action

            if phase != last_phase:
                phase_log.append({"step": step, "phase": phase})
                last_phase = phase

            gripper_sign = 1 if float(action[0, -1].detach().cpu().item()) >= 0.0 else -1
            if last_gripper_sign is None:
                gripper_command_events.append({"step": step, "phase": phase, "command": gripper_sign})
            elif gripper_sign != last_gripper_sign:
                gripper_command_transitions += 1
                gripper_command_events.append({"step": step, "phase": phase, "command": gripper_sign})
            last_gripper_sign = gripper_sign

            obs, _, _, _, _ = env.step(action)
            controller.after_step()
            controller.assist_object(env, obs["policy"])
            controller.tick_phase()

            object_quat_error = quaternion_angle_error(controller.object_cmd_quat_w, object_quat_ref_w)
            object_quat_max_angle_error_rad = max(
                object_quat_max_angle_error_rad,
                float(object_quat_error.max().detach().cpu().item()),
            )
            gripper_pos = obs["policy"].get("gripper_pos")
            if gripper_pos is not None:
                gripper_scalar = float(gripper_pos.reshape(gripper_pos.shape[0], -1).mean().detach().cpu().item())
                gripper_pos_min = gripper_scalar if gripper_pos_min is None else min(gripper_pos_min, gripper_scalar)
                gripper_pos_max = gripper_scalar if gripper_pos_max is None else max(gripper_pos_max, gripper_scalar)
                if gripper_pos_prev is not None:
                    gripper_pos_max_step = max(gripper_pos_max_step, abs(gripper_scalar - gripper_pos_prev))
                gripper_pos_prev = gripper_scalar
            if spec.base.label == "galbot_cube" and args.show_cube_frame_markers:
                cube_frame_markers_enabled = (
                    update_cube_frame_markers(env, controller.object_cmd_pos_w, controller.object_cmd_quat_w)
                    or cube_frame_markers_enabled
                )

            if args.record_frames and step % 2 == 0:
                env.sim.render()
                frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                frame_ok = False
                if args.capture_backend in ("auto", "camera"):
                    frame_ok = save_camera_frame(env, frame_path, env_step_s)
                if not frame_ok and viewport_capture is not None:
                    frame_ok = viewport_capture.save_frame(env, frame_path)
                if not frame_ok:
                    print(f"WARNING: recorded frame appears black: {frame_path}", flush=True)
                frame_count += 1

            command_target_error = float(
                torch.linalg.vector_norm(controller.object_cmd_pos_w[:, :3] - target_w_for_success[:, :3], dim=-1)
                .max()
                .item()
            )
            if controller.released and command_target_error < 0.025:
                done_step = step
                command_success = True
                break
            if controller.done and mobile_rollout_success(env, spec):
                done_step = step
                break

    robot = env.scene["robot"]
    object_pos = env.scene[spec.base.object_name].data.root_pos_w[0].detach().cpu().tolist()
    object_quat = env.scene[spec.base.object_name].data.root_quat_w[0].detach().cpu().tolist()
    object_cmd_quat = controller.object_cmd_quat_w[0].detach().cpu().tolist()
    base_final = robot.data.root_pos_w[0].detach().cpu().tolist()
    final_eef_b = obs["policy"]["eef_pos"][0].detach().cpu().tolist()
    final_eef_w = base_to_world(env, obs["policy"]["eef_pos"].reshape(env.num_envs, 3))[0].detach().cpu().tolist()
    final_target_w = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    final_target_b = world_to_base(env, final_target_w)[0].detach().cpu().tolist()
    final_target_error_m = float(
        torch.linalg.vector_norm(
            env.scene[spec.base.object_name].data.root_pos_w[:, :3] - final_target_w[:, :3], dim=-1
        )
        .max()
        .item()
    )
    command_target_error_m = float(
        torch.linalg.vector_norm(controller.object_cmd_pos_w[:, :3] - final_target_w[:, :3], dim=-1).max().item()
    )
    rollout_success = mobile_rollout_success(env, spec)
    demo_success = bool(rollout_success or command_success)
    joint_names = list(getattr(robot, "joint_names", []))
    wheel_like_joints = [
        name
        for name in joint_names
        if any(token in name.lower() for token in ("wheel", "caster", "drive", "tire"))
    ]
    summary = {
        "task": spec.base.task,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "success": demo_success,
        "rollout_success": rollout_success,
        "command_success": command_success,
        "done_step": done_step,
        "final_phase": controller.phase_name,
        "controller_done": controller.done,
        "object_name": spec.base.object_name,
        "object_motion_mode": "bounded_interpolation_no_instant_snap",
        "object_pos_w": object_pos,
        "object_quat_w": object_quat,
        "object_command_quat_w": object_cmd_quat,
        "object_locked_quat_w": object_quat_ref_w[0].detach().cpu().tolist(),
        "object_quat_max_angle_error_rad": object_quat_max_angle_error_rad,
        "object_quat_max_angle_error_deg": math.degrees(object_quat_max_angle_error_rad),
        "object_target_error_m": final_target_error_m,
        "object_command_target_error_m": command_target_error_m,
        "object_max_step_m": controller.max_object_step_m,
        "object_release_settle_steps": controller.release_settle_steps,
        "final_eef_pos_b": final_eef_b,
        "final_eef_pos_w": final_eef_w,
        "final_target_pos_b": final_target_b,
        "target_pos_w": list(spec.target_world_pos),
        "source_table_pos_w": list(spec.source_table_pos),
        "source_object_pose_w": source_object_pose,
        "source_standoff_pose_w": source_standoff_pose,
        "target_table_pos_w": list(spec.target_table_pos),
        "target_table_scale": list(spec.target_table_scale),
        "table_distance_m": table_distance,
        "base_start_pos_w": controller.root_start_pos[0].detach().cpu().tolist(),
        "base_goal_pos_w": controller.root_goal_pos[0].detach().cpu().tolist(),
        "base_final_pos_w": base_final,
        "base_curve_points_w": controller.route_waypoints[0].detach().cpu().tolist(),
        "base_curve_sample_count": int(curve_samples.shape[0]),
        "base_direct_displacement_m": base_direct_displacement,
        "base_travel_m": base_route_length,
        "base_motion_mode": "kinematic_articulation_root_nonholonomic_turn_drive_turn_collision_checked",
        "base_segment_drive_yaws_rad": controller.segment_drive_yaws[0].detach().cpu().tolist(),
        "docking_yaw_rad": float(controller.root_goal_yaw[0].detach().cpu().item()),
        "cube_frame_markers_enabled": cube_frame_markers_enabled,
        "hidden_cube_names": hidden_cube_names,
        "gripper_command_transitions": gripper_command_transitions,
        "gripper_command_events": gripper_command_events,
        "gripper_pos_range": None if gripper_pos_min is None else [gripper_pos_min, gripper_pos_max],
        "gripper_pos_max_step": gripper_pos_max_step,
        "collision_check_passed": controller.collision_check_passed,
        "collision_clearance_threshold_m": controller.clearance_threshold_m,
        "decor_collision_enabled": False,
        "docking_pose_w": controller.root_goal_pos[0].detach().cpu().tolist(),
        "drive_min_table_center_clearance_m": drive_min_table_center_clearance,
        "min_table_clearance_m": controller.min_table_clearance_m,
        "robot_base_radius_m": controller.robot_base_radius,
        "route_waypoints_w": controller.route_waypoints[0].detach().cpu().tolist(),
        "scene_style": "isaac_warehouse_props",
        "table_aabbs_xy": table_aabbs_xy,
        "wheel_like_joints_exposed": wheel_like_joints,
        "camera_resolution": [QHD_WIDTH, QHD_HEIGHT] if args.record_frames else None,
        "camera_quality": "qhd_2560x1440_zoomed",
        "frames": frame_count,
        "frames_dir": str(frames_dir.resolve()) if args.record_frames else None,
        "phase_log": phase_log,
    }
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
