from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from pickplace_common import (
    MobileTaskSpec,
    TaskSpec,
    add_isaac_warehouse_scene,
    add_mobile_demo_camera,
    add_mobile_target_table,
    base_to_world,
    configure_mobile_source_table,
    hold_current_pose_action,
    set_rigid_pose,
    set_robot_root_pose,
    world_to_base,
)
from play_mobile_pickplace import (
    get_table_aabbs_xy,
    image_has_content,
    lerp_yaw,
    point_to_aabb_clearance,
    quat_to_yaw,
    route_clearance,
    save_camera_frame,
    smooth_step,
    wrap_angle,
    yaw_to_quat,
)

TASK = "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0"
LABEL = "agibot_a2d_cube"
SOURCE_TABLE_POS = (1.10, 0.0, 0.60)
TARGET_TABLE_POS = (4.80, 0.0, 0.60)
TABLE_SCALE = (1.0, 1.0, 0.60)
CUBE_SCALE = 2.0
CUBE_HALF_HEIGHT = 0.0203 * CUBE_SCALE
CUBE_SOURCE_POS = (0.39, -0.18, SOURCE_TABLE_POS[2] + CUBE_HALF_HEIGHT)
CUBE_TARGET_POS = (4.09, -0.18, TARGET_TABLE_POS[2] + CUBE_HALF_HEIGHT)
GRASP_EEF_X_OFFSET_M = -0.025
GRASP_EEF_Y_OFFSET_M = -0.055
GRASP_EEF_Z_OFFSET = 0.040
APPROACH_EEF_Z_OFFSET = 0.220
RETREAT_EEF_Z_OFFSET = 0.260
CARRY_OBJECT_LIFT_M = 0.145
CARRY_OBJECT_X_B = 0.74
CARRY_OBJECT_Y_B = -0.14
AISLE_Y = -1.60
START_DOCK_X_SHIFT_M = 0.11
ROBOT_BASE_RADIUS_M = 0.42
TABLE_CLEARANCE_THRESHOLD_M = 0.30
STATIC_OBSTACLE_CLEARANCE_THRESHOLD_M = 0.20
CUBE_TABLE_HEIGHT_TOLERANCE_M = 0.008
PICK_XY_TOLERANCE_M = 0.16
ATTACHED_SYNC_ERROR_THRESHOLD_M = 0.015
ATTACHED_OBJECT_MAX_STEP_M = 0.085
CARRIED_EEF_STEP_M = 0.035
PRECLAMP_OBJECT_STEP_M = 0.006
ATTACHED_OFFSET_BLEND_STEP_M = 0.004
OFFSET_Z_BLEND_STEP_M = 0.0015
CARRY_GRASP_VISUAL_OFFSET_B = (0.0, 0.0, 0.0)
OBJECT_TELEPORT_STEP_THRESHOLD_M = 0.13
BODY_KEEP_OUT_RADIUS_M = 0.58
POST_RELEASE_SETTLE_STEPS = 72
GRIPPER_CAPTURE_ERROR_THRESHOLD_M = 0.035
GRIPPER_CENTERED_GRASP_OFFSET_B = (0.0, 0.0, 0.028)
GRIPPER_OPEN_JOINT_POS = 0.994
GRIPPER_CUBE_HOLD_JOINT_POS = 0.18
LIFT_DRIVEN_PICK_PHASES = {"lift_up_carry", "hold_carry"}
LIFT_DRIVEN_PLACE_PHASES = {"lower_lift_place", "open"}
QHD_WIDTH = 2560
QHD_HEIGHT = 1440


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agibot A2D lift-assisted cube pick and place demo.")
    parser.add_argument("--max_steps", type=int, default=3400)
    parser.add_argument("--drive_steps", type=int, default=1800)
    parser.add_argument("--record_frames", action="store_true")
    parser.add_argument("--frames_dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--capture_every", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def make_spec() -> MobileTaskSpec:
    base = TaskSpec(
        task=TASK,
        label=LABEL,
        object_name="mug",
        object_pos_key="mug_positions",
        target_world_pos=CUBE_SOURCE_POS,
        target_table_pos=SOURCE_TABLE_POS,
        target_table_scale=TABLE_SCALE,
        grasp_z=GRASP_EEF_Z_OFFSET,
        approach_z=APPROACH_EEF_Z_OFFSET,
        lift_z=APPROACH_EEF_Z_OFFSET,
        place_z=GRASP_EEF_Z_OFFSET,
        hold_offset=(0.0, 0.0, -GRASP_EEF_Z_OFFSET),
        carry_quat_world=(1.0, 0.0, 0.0, 0.0),
    )
    return MobileTaskSpec(
        base=base,
        source_table_pos=SOURCE_TABLE_POS,
        target_world_pos=CUBE_TARGET_POS,
        target_table_pos=TARGET_TABLE_POS,
        target_table_scale=TABLE_SCALE,
    )


def configure_cube_object(env_cfg) -> None:
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=100.0,
        max_linear_velocity=100.0,
        max_depenetration_velocity=2.0,
        disable_gravity=False,
    )
    env_cfg.scene.mug.init_state.pos = list(CUBE_SOURCE_POS)
    env_cfg.scene.mug.init_state.rot = [1.0, 0.0, 0.0, 0.0]
    env_cfg.scene.mug.spawn = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
        scale=(CUBE_SCALE, CUBE_SCALE, CUBE_SCALE),
        rigid_props=cube_properties,
        collision_props=CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        mass_props=MassPropertiesCfg(mass=0.05),
    )

    gripper_action = getattr(getattr(env_cfg, "actions", None), "gripper_action", None)
    if gripper_action is not None and hasattr(gripper_action, "close_command_expr"):
        gripper_action.close_command_expr = {"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0}

    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None


def set_camera_view(env, env_step_s: float) -> None:
    eye = torch.tensor([[2.92, -8.10, 2.78]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    lookat = torch.tensor([[3.05, -0.30, 0.82]], dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    env.scene["demo_camera"].set_world_poses_from_view(eye, lookat)
    for _ in range(4):
        env.sim.render()
        env.scene["demo_camera"].update(env_step_s, force_recompute=True)


def record_camera_frame(env, frame_path: Path, env_step_s: float, previous_frame_path: Path | None) -> bool:
    for _ in range(4):
        env.sim.render()
        if save_camera_frame(env, frame_path, env_step_s) and image_has_content(frame_path):
            return True
    if previous_frame_path is not None and previous_frame_path.is_file():
        shutil.copyfile(previous_frame_path, frame_path)
        return True
    return image_has_content(frame_path)


def table_penetration_clearance(
    point_xy: torch.Tensor,
    table_aabbs_xy: dict[str, dict[str, list[float]]],
    robot_radius: float,
) -> float:
    if not table_aabbs_xy:
        return float("inf")
    clearances = [point_to_aabb_clearance(point_xy, aabb, robot_radius) for aabb in table_aabbs_xy.values()]
    return float(torch.stack(clearances, dim=-1).min().detach().cpu().item())


def get_static_obstacle_aabbs_xy() -> dict[str, dict[str, list[float]]]:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out: dict[str, dict[str, list[float]]] = {}
    include_tokens = ("WarehouseAsset", "WarehouseTrim")
    ignore_tokens = (
        "floor",
        "wall",
        "lane",
        "light",
        "mat",
        "belt",
        "fence",
        "monitor",
        "bench",
        "conveyor",
    )
    env_prim = stage.GetPrimAtPath("/World/envs/env_0")
    if not env_prim.IsValid():
        return out
    for prim in Usd.PrimRange(env_prim):
        path = str(prim.GetPath())
        leaf = path.rsplit("/", 1)[-1]
        lower = leaf.lower()
        if not any(token in leaf for token in include_tokens):
            continue
        if any(token in lower for token in ignore_tokens):
            continue
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = bbox.GetMin()
        mx = bbox.GetMax()
        width = float(mx[0] - mn[0])
        depth = float(mx[1] - mn[1])
        height = float(mx[2] - mn[2])
        if width <= 0.02 or depth <= 0.02 or height <= 0.05:
            continue
        if width > 2.8 or depth > 2.8:
            continue
        out[leaf] = {"min": [float(mn[0]), float(mn[1])], "max": [float(mx[0]), float(mx[1])]}
    return out


def height_error_on_table(pos_w: torch.Tensor, table_z: float) -> float:
    expected = table_z + CUBE_HALF_HEIGHT
    return float(torch.abs(pos_w[:, 2] - expected).max().detach().cpu().item())


def world_to_base_with_pose(pos_w: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device)
    quat_w[:, 0] = 1.0
    pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)
    return pos_b


class AgibotA2DCubeController:
    base_motion_phases = ("undock", "aisle_drive", "dock")
    phase_names = [
        "approach",
        "lift_down_pick",
        "descend",
        "close",
        "lift_up_carry",
        "hold_carry",
        "undock",
        "aisle_drive",
        "dock",
        "transfer_align",
        "lower_lift_place",
        "open",
        "retreat",
    ]

    def __init__(
        self,
        spec: MobileTaskSpec,
        env,
        initial_policy_obs: dict[str, torch.Tensor],
        drive_steps: int,
        table_aabbs_xy: dict[str, dict[str, list[float]]],
    ):
        self.spec = spec
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.action_dim = env.action_space.shape[-1]
        self.drive_steps = max(480, drive_steps)
        self.phase_index = 0
        self.phase_step_count = 0
        self.motion_phase_count = 0
        self.hold_count = 0
        self.attached = False
        self.released = False
        self.object_max_step_m = 0.0
        self.object_max_step_phase: str | None = None
        self.object_max_step_from_w: list[float] | None = None
        self.object_max_step_to_w: list[float] | None = None
        self.pregrasp_max_object_motion_m = 0.0
        self.attached_sync_error_max_m = 0.0
        self.attached_sync_error_last_m = 0.0
        self.gripper_object_sync_error_max_m = 0.0
        self.gripper_object_sync_error_last_m = 0.0
        self.preclamp_max_object_step_m = 0.0
        self.preclamp_final_error_m = float("inf")
        self.gripper_capture_error_m = float("inf")
        self.gripper_capture_offset_b: list[float] | None = None
        self.gripper_capture_anchor_w: list[float] | None = None
        self.gripper_capture_anchor_quat_w: list[float] | None = None
        self.gripper_capture_center_w: list[float] | None = None
        self.gripper_capture_pad_positions_w: list[list[float]] | None = None
        self.gripper_capture_pad_gap_m: float | None = None
        self.gripper_capture_body_positions_w: dict[str, list[float]] | None = None
        self.gripper_attach_snap_m = float("inf")
        self.min_carried_object_body_clearance_m = float("inf")
        self.min_carried_gripper_body_clearance_m = float("inf")
        self.release_settle_steps = 0
        self.release_pose_w: list[float] | None = None
        self.table_aabbs_xy = table_aabbs_xy
        self.robot_base_radius = ROBOT_BASE_RADIUS_M
        self.clearance_threshold_m = TABLE_CLEARANCE_THRESHOLD_M

        robot = env.scene["robot"]
        self.root_start_pos = robot.data.root_pos_w.clone()
        self.root_start_quat = robot.data.root_quat_w.clone()
        self.root_start_pos[:, 0] += START_DOCK_X_SHIFT_M
        set_robot_root_pose(self.env, self.root_start_pos, self.root_start_quat)
        self.root_start_yaw = quat_to_yaw(self.root_start_quat)
        self.root_goal_pos = self.root_start_pos.clone()
        table_delta = torch.tensor(
            [
                TARGET_TABLE_POS[0] - SOURCE_TABLE_POS[0],
                TARGET_TABLE_POS[1] - SOURCE_TABLE_POS[1],
                0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.root_goal_pos += table_delta.repeat(self.num_envs, 1)
        self.root_goal_yaw = self.root_start_yaw.clone()
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
        self.route_samples_xy = torch.empty((0, 2), dtype=torch.float32, device=self.device)
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

        object_asset = env.scene[self.spec.base.object_name]
        self.source_object_pos_w = object_asset.data.root_pos_w.clone()
        self.source_object_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.source_object_quat_w[:, 0] = 1.0
        self.object_cmd_pos_w = self.source_object_pos_w.clone()
        self.object_cmd_quat_w = self.source_object_quat_w.clone()
        self.object_start_b = world_to_base_with_pose(
            self.source_object_pos_w, self.root_start_pos, self.root_start_quat
        )
        self.grasp_offset_b = torch.tensor(
            [[GRASP_EEF_X_OFFSET_M, -GRASP_EEF_Y_OFFSET_M, -GRASP_EEF_Z_OFFSET]],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.attached_offset_b = self._z(-GRASP_EEF_Z_OFFSET)
        self.attached_offset_static_b: torch.Tensor | None = None
        self.attached_offset_goal_b: torch.Tensor | None = None
        self.attached_offset_eef_b: torch.Tensor | None = None
        self.attached_pad_offset_b: torch.Tensor | None = None
        self.centered_grasp_offset_b = torch.tensor(
            [GRIPPER_CENTERED_GRASP_OFFSET_B],
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.attached_visual_offset_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.attached_offset_locked = False
        self.attach_offset_capture_step: int | None = None
        self.carried_eef_b: torch.Tensor | None = None
        self.carry_object_b = self.object_start_b.clone()
        self.carry_object_b[:, 0] = CARRY_OBJECT_X_B
        self.carry_object_b[:, 1] = CARRY_OBJECT_Y_B
        self.carry_object_b[:, 2:3] = self.object_start_b[:, 2:3] + CARRY_OBJECT_LIFT_M
        self.carry_eef_b = self.eef_for_object(self.carry_object_b)
        self.current_eef_target_b = self.eef_for_object(self.object_start_b, APPROACH_EEF_Z_OFFSET)

        lift_ids, _ = robot.find_joints(["joint_lift_body"], preserve_order=True)
        body_pitch_ids, _ = robot.find_joints(["joint_body_pitch"], preserve_order=True)
        gripper_hold_ids, gripper_hold_names = robot.find_joints(["left_hand_joint1"], preserve_order=False)
        gripper_center_ids, gripper_center_names = robot.find_bodies(["gripper_center"], preserve_order=False)
        left_pad_ids, left_pad_names = robot.find_bodies(["left_.*_Pad_Link"], preserve_order=False)
        gripper_probe_ids, gripper_probe_names = robot.find_bodies(
            ["left_base_link", "left_.*_Link", "gripper_center"],
            preserve_order=False,
        )
        self.lift_joint_ids = lift_ids
        self.body_pitch_joint_ids = body_pitch_ids
        self.gripper_hold_joint_ids = gripper_hold_ids
        self.gripper_hold_joint_names = list(gripper_hold_names)
        self.gripper_center_body_ids = gripper_center_ids
        self.gripper_center_body_names = list(gripper_center_names)
        self.left_pad_body_ids = left_pad_ids
        self.left_pad_body_names = list(left_pad_names)
        self.gripper_probe_body_ids = gripper_probe_ids
        self.gripper_probe_body_names = list(gripper_probe_names)
        self.lift_low = 0.135
        self.lift_mid = 0.190
        self.lift_high = 0.225
        self.lift_max_step = 0.0010
        self.lift_trace: list[dict[str, float | int | str]] = []
        self.lift_min = float("inf")
        self.lift_max = float("-inf")
        self.initial_eef_b = initial_policy_obs["eef_pos"][0].detach().cpu().tolist()

    @staticmethod
    def _route_segment_yaws(route_waypoints: torch.Tensor) -> torch.Tensor:
        deltas = route_waypoints[:, 1:, :2] - route_waypoints[:, :-1, :2]
        return torch.atan2(deltas[:, :, 1], deltas[:, :, 0])

    @staticmethod
    def _split_drive_steps(drive_steps: int) -> list[int]:
        undock = max(20, int(round(float(drive_steps) * 0.23)))
        dock = max(20, int(round(float(drive_steps) * 0.23)))
        aisle = max(30, drive_steps - undock - dock)
        return [undock, aisle, dock]

    def _z(self, value: float) -> torch.Tensor:
        out = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        out[:, 2] = value
        return out

    def target_object_b(self) -> torch.Tensor:
        target_w = torch.tensor(CUBE_TARGET_POS, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        return world_to_base(self.env, target_w)

    def lift_joint_value(self) -> torch.Tensor:
        if not self.lift_joint_ids:
            return torch.full((self.num_envs, 1), self.lift_high, dtype=torch.float32, device=self.device)
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[:, self.lift_joint_ids].reshape(self.num_envs, -1).mean(dim=1, keepdim=True)

    def pick_lift_progress(self) -> torch.Tensor:
        denom = max(self.lift_high - self.lift_low, 1.0e-6)
        raw = torch.clamp((self.lift_joint_value() - self.lift_low) / denom, min=0.0, max=1.0)
        return smooth_step(raw)

    def place_lift_progress(self) -> torch.Tensor:
        denom = max(self.lift_high - self.lift_mid, 1.0e-6)
        raw = torch.clamp((self.lift_joint_value() - self.lift_mid) / denom, min=0.0, max=1.0)
        return smooth_step(raw)

    def pick_lift_object_b(self) -> torch.Tensor:
        progress = self.pick_lift_progress()
        object_b = self.object_start_b.clone()
        object_b[:, :2] = self.object_start_b[:, :2] + (self.carry_object_b[:, :2] - self.object_start_b[:, :2]) * progress
        object_b[:, 2:3] = self.object_start_b[:, 2:3] + CARRY_OBJECT_LIFT_M * progress
        return object_b

    def target_lift_object_b(self) -> torch.Tensor:
        object_b = self.target_object_b()
        object_b[:, 2:3] = object_b[:, 2:3] + CARRY_OBJECT_LIFT_M * self.place_lift_progress()
        return object_b

    def eef_quat_from_policy(self, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        eef_quat_b = policy_obs["eef_quat"].reshape(self.num_envs, 4).clone()
        return eef_quat_b / torch.linalg.vector_norm(eef_quat_b, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def attached_offset_from_policy(
        self, policy_obs: dict[str, torch.Tensor] | None = None, include_visual: bool = True
    ) -> torch.Tensor:
        if not self.attached_offset_locked:
            return self.grasp_offset_b
        if policy_obs is None or self.attached_offset_eef_b is None:
            return self.attached_offset_b
        import isaaclab.utils.math as math_utils

        offset_b = math_utils.quat_apply(self.eef_quat_from_policy(policy_obs), self.attached_offset_eef_b)
        if include_visual:
            offset_b = offset_b + self.attached_visual_offset_b
        return offset_b

    def eef_for_object(
        self,
        object_b: torch.Tensor,
        z_offset: float | None = None,
        policy_obs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        offset_b = self.attached_offset_from_policy(policy_obs, include_visual=False)
        target = object_b - offset_b
        if z_offset is not None:
            target[:, 2] = object_b[:, 2] + z_offset
        return target

    def eef_for_attached_object(
        self, object_b: torch.Tensor, policy_obs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        eef_b = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        current_object_b = self.attached_object_pos_b(eef_b, policy_obs)
        return eef_b + (object_b - current_object_b)

    def eef_for_gripper_anchor(
        self,
        anchor_target_b: torch.Tensor,
        policy_obs: dict[str, torch.Tensor],
        z_offset: float | None = None,
    ) -> torch.Tensor:
        target = anchor_target_b.clone()
        if z_offset is not None:
            target[:, 2] = anchor_target_b[:, 2] + z_offset
        eef_b = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        current_anchor_b = self.gripper_anchor_b(policy_obs)
        return eef_b + (target - current_anchor_b)

    def eef_for_centered_grasp(
        self,
        object_b: torch.Tensor,
        policy_obs: dict[str, torch.Tensor],
        z_offset: float | None = None,
    ) -> torch.Tensor:
        target_object_b = object_b.clone()
        if z_offset is not None:
            target_object_b[:, 2] = object_b[:, 2] + z_offset
        anchor_target_b = target_object_b - self.centered_grasp_offset_b
        return self.eef_for_gripper_anchor(anchor_target_b, policy_obs)

    def grasp_point_for_eef(
        self, eef_b: torch.Tensor, policy_obs: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        offset_b = self.attached_offset_from_policy(policy_obs, include_visual=True)
        return eef_b + offset_b

    def left_gripper_pad_midpoint_b(self) -> torch.Tensor | None:
        pad_pos_w = self.left_gripper_pad_midpoint_w()
        if pad_pos_w is None:
            return None
        return world_to_base(self.env, pad_pos_w)

    def left_gripper_pad_midpoint_w(self) -> torch.Tensor | None:
        if not self.left_pad_body_ids:
            return None
        robot = self.env.scene["robot"]
        return robot.data.body_pos_w[:, self.left_pad_body_ids, :].mean(dim=1)

    def left_gripper_center_b(self) -> torch.Tensor | None:
        center_pos_w = self.left_gripper_center_w()
        if center_pos_w is None:
            return None
        return world_to_base(self.env, center_pos_w)

    def left_gripper_center_w(self) -> torch.Tensor | None:
        if not self.gripper_center_body_ids:
            return None
        robot = self.env.scene["robot"]
        return robot.data.body_pos_w[:, self.gripper_center_body_ids, :].mean(dim=1)

    def gripper_anchor_w(self) -> torch.Tensor | None:
        pad_midpoint_w = self.left_gripper_pad_midpoint_w()
        if pad_midpoint_w is not None:
            return pad_midpoint_w
        return self.left_gripper_center_w()

    def gripper_anchor_quat_w(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        if self.gripper_center_body_ids:
            quat_w = robot.data.body_quat_w[:, self.gripper_center_body_ids[0], :]
        elif self.left_pad_body_ids:
            quat_w = robot.data.body_quat_w[:, self.left_pad_body_ids[0], :]
        else:
            quat_w = robot.data.root_quat_w
        return quat_w / torch.linalg.vector_norm(quat_w, dim=-1, keepdim=True).clamp_min(1.0e-6)

    def gripper_anchor_b(self, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor_w = self.gripper_anchor_w()
        if anchor_w is not None:
            return world_to_base(self.env, anchor_w)
        return policy_obs["eef_pos"].reshape(self.num_envs, 3)

    def attached_object_pos_b(self, eef_b: torch.Tensor, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.attached_pad_offset_b is not None:
            return self.gripper_anchor_b(policy_obs) + self.attached_pad_offset_b
        return self.grasp_point_for_eef(eef_b, policy_obs)

    def gripper_offset_to_world(self, offset_b: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        robot = self.env.scene["robot"]
        return math_utils.quat_apply(robot.data.root_quat_w, offset_b)

    def gripper_local_offset_to_world(self, offset_local: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        return math_utils.quat_apply(self.gripper_anchor_quat_w(), offset_local)

    def attached_object_pos_w(self, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor_w = self.gripper_anchor_w()
        if anchor_w is not None and self.attached_pad_offset_b is not None:
            return anchor_w + self.gripper_offset_to_world(self.attached_pad_offset_b)
        eef_b = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        return base_to_world(self.env, self.attached_object_pos_b(eef_b, policy_obs))

    def centered_grasp_object_pos_w(self, policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        anchor_w = self.gripper_anchor_w()
        if anchor_w is not None:
            return anchor_w + self.gripper_offset_to_world(self.centered_grasp_offset_b)
        anchor_b = self.gripper_anchor_b(policy_obs)
        return base_to_world(self.env, anchor_b + self.centered_grasp_offset_b)

    def update_attached_offset(self, policy_obs: dict[str, torch.Tensor]) -> None:
        if not self.attached_offset_locked:
            return
        desired_visual = torch.zeros_like(self.attached_visual_offset_b)
        if self.phase_name in ("undock", "aisle_drive"):
            desired_visual = torch.tensor(
                [CARRY_GRASP_VISUAL_OFFSET_B], dtype=torch.float32, device=self.device
            ).repeat(self.num_envs, 1)
        visual_delta = desired_visual - self.attached_visual_offset_b
        visual_dist = torch.linalg.vector_norm(visual_delta, dim=-1, keepdim=True)
        visual_ratio = torch.clamp(ATTACHED_OFFSET_BLEND_STEP_M / (visual_dist + 1.0e-6), max=1.0)
        self.attached_visual_offset_b = self.attached_visual_offset_b + visual_delta * visual_ratio
        self.attached_offset_b = self.attached_offset_from_policy(policy_obs)

    def lift_target(self) -> float:
        phase = self.phase_name
        if phase in (
            "approach",
            "lift_up_carry",
            "hold_carry",
            "undock",
            "aisle_drive",
            "dock",
            "transfer_align",
            "retreat",
        ):
            return self.lift_high
        if phase in ("lift_down_pick", "descend", "close"):
            return self.lift_low
        if phase in ("lower_lift_place", "open"):
            return self.lift_mid
        return self.lift_mid

    def phase_target(self, policy_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float, int, bool, bool]:
        open_cmd = 1.0
        close_cmd = -1.0
        phase = self.phase_name
        if phase == "approach":
            return self.eef_for_centered_grasp(self.object_start_b, policy_obs, APPROACH_EEF_Z_OFFSET), open_cmd, 0, False, False
        if phase == "lift_down_pick":
            return self.eef_for_centered_grasp(self.object_start_b, policy_obs, 0.150), open_cmd, 105, False, False
        if phase == "descend":
            return self.eef_for_centered_grasp(self.object_start_b, policy_obs), open_cmd, 0, False, False
        if phase == "close":
            return self.eef_for_centered_grasp(self.object_start_b, policy_obs), close_cmd, 120, True, False
        if phase == "lift_up_carry":
            return self.eef_for_attached_object(self.pick_lift_object_b(), policy_obs), close_cmd, 0, True, False
        if phase == "hold_carry":
            return self.eef_for_attached_object(self.pick_lift_object_b(), policy_obs), close_cmd, 120, True, False
        target_b = self.target_object_b()
        if phase == "transfer_align":
            return self.eef_for_attached_object(target_b + self._z(CARRY_OBJECT_LIFT_M), policy_obs), close_cmd, 24, True, False
        if phase == "lower_lift_place":
            return self.eef_for_attached_object(self.target_lift_object_b(), policy_obs), close_cmd, 0, True, False
        if phase == "open":
            return self.eef_for_attached_object(target_b, policy_obs), open_cmd, 38, False, True
        return self.eef_for_object(target_b, RETREAT_EEF_Z_OFFSET), open_cmd, 0, False, True

    def command_lift(self, step: int) -> None:
        if not self.lift_joint_ids:
            return
        robot = self.env.scene["robot"]
        current = robot.data.joint_pos[:, self.lift_joint_ids].clone()
        target = torch.full_like(current, self.lift_target())
        delta = torch.clamp(target - current, min=-self.lift_max_step, max=self.lift_max_step)
        new_position = current + delta
        zero_velocity = torch.zeros_like(new_position)
        robot.write_joint_state_to_sim(new_position, zero_velocity, joint_ids=self.lift_joint_ids)
        robot.set_joint_position_target(new_position, joint_ids=self.lift_joint_ids)
        if self.body_pitch_joint_ids:
            pitch_target = robot.data.default_joint_pos[:, self.body_pitch_joint_ids].clone()
            robot.set_joint_position_target(pitch_target, joint_ids=self.body_pitch_joint_ids)
        value = float(new_position[0, 0].detach().cpu().item())
        self.lift_min = min(self.lift_min, value)
        self.lift_max = max(self.lift_max, value)
        if step % 10 == 0 or self.phase_step_count == 0:
            self.lift_trace.append({"step": step, "phase": self.phase_name, "joint_lift_body": value})

    def command_gripper_aperture(self) -> None:
        if not self.gripper_hold_joint_ids:
            return
        if self.phase_name in ("close", "lift_up_carry", "hold_carry", "undock", "aisle_drive", "dock", "transfer_align", "lower_lift_place"):
            target_value = GRIPPER_CUBE_HOLD_JOINT_POS
        elif self.attached:
            target_value = GRIPPER_CUBE_HOLD_JOINT_POS
        else:
            return
        robot = self.env.scene["robot"]
        target = torch.full(
            (self.num_envs, len(self.gripper_hold_joint_ids)),
            target_value,
            dtype=torch.float32,
            device=self.device,
        )
        zero_velocity = torch.zeros_like(target)
        robot.write_joint_state_to_sim(target, zero_velocity, joint_ids=self.gripper_hold_joint_ids)
        robot.set_joint_position_target(target, joint_ids=self.gripper_hold_joint_ids)

    def act(self, policy_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
        self.update_attached_offset(policy_obs)
        if self.phase_name in self.base_motion_phases:
            target = self.eef_for_attached_object(self.carry_object_b, policy_obs)
            self.current_eef_target_b = target.clone()
            return self.arm_action_to_target(policy_obs, target, gripper=-1.0, max_delta=0.035), self.phase_name

        target, gripper, hold_steps, attach, release = self.phase_target(policy_obs)
        self.current_eef_target_b = target.clone()
        action = self.arm_action_to_target(policy_obs, target, gripper=gripper)

        eef = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        dist = torch.linalg.vector_norm(target - eef, dim=-1)

        if release:
            if self.release_pose_w is None:
                self.release_pose_w = self.object_cmd_pos_w[0].detach().cpu().tolist()
            self.attached = False
            self.released = True
        elif attach and hold_steps == 0:
            self.attached = True

        if hold_steps > 0:
            self.hold_count += 1
            if self.hold_count >= hold_steps:
                if attach:
                    self.capture_attached_offset(policy_obs)
                    self.attached = True
                self.hold_count = 0
                self.advance_phase()
        else:
            if self.phase_name == "lower_lift_place":
                tolerance = 0.006
            elif self.phase_name == "descend":
                tolerance = 0.030
            else:
                tolerance = 0.045
            if bool((dist < tolerance).all().item()):
                self.hold_count = 0
                self.advance_phase()
        return action, self.phase_name

    def arm_action_to_target(
        self,
        policy_obs: dict[str, torch.Tensor],
        target: torch.Tensor,
        gripper: float,
        max_delta: float | None = None,
    ) -> torch.Tensor:
        eef = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        delta = target - eef
        dist = torch.linalg.vector_norm(delta, dim=-1)
        action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        if max_delta is None:
            max_delta = 0.028 if self.phase_name in ("descend", "lower_lift_place") else 0.045
        scale = torch.clamp(float(max_delta) / (dist + 1.0e-6), max=1.0).unsqueeze(-1)
        action[:, :3] = delta * scale
        action[:, -1] = gripper
        return action

    def advance_phase(self) -> None:
        self.phase_index += 1
        self.phase_step_count = 0
        self.motion_phase_count = 0

    def tick_phase(self) -> None:
        self.phase_step_count += 1
        max_steps = {
            "approach": 150,
            "lift_down_pick": 140,
            "descend": 170,
            "lift_up_carry": 145,
            "hold_carry": 150,
            "transfer_align": 130,
            "lower_lift_place": 420,
            "retreat": 110,
        }
        if self.phase_name in max_steps and self.phase_step_count >= max_steps[self.phase_name]:
            self.advance_phase()

    def capture_attached_offset(self, policy_obs: dict[str, torch.Tensor]) -> None:
        if self.attached_offset_locked:
            return
        eef_b = policy_obs["eef_pos"].reshape(self.num_envs, 3).clone()
        object_b = world_to_base(self.env, self.object_cmd_pos_w)
        self.attached_offset_b = object_b - eef_b
        self.attached_offset_static_b = self.attached_offset_b.clone()
        import isaaclab.utils.math as math_utils

        anchor_w = self.gripper_anchor_w()
        if anchor_w is not None:
            self.gripper_capture_anchor_w = anchor_w[0].detach().cpu().tolist()
            self.gripper_capture_anchor_quat_w = self.gripper_anchor_quat_w()[0].detach().cpu().tolist()
            center_w = self.left_gripper_center_w()
            if center_w is not None:
                self.gripper_capture_center_w = center_w[0].detach().cpu().tolist()
            if self.left_pad_body_ids:
                robot = self.env.scene["robot"]
                pad_pos_w = robot.data.body_pos_w[:, self.left_pad_body_ids, :]
                self.gripper_capture_pad_positions_w = pad_pos_w[0].detach().cpu().tolist()
                if pad_pos_w.shape[1] >= 2:
                    pad_gap = torch.linalg.vector_norm(pad_pos_w[:, 0, :] - pad_pos_w[:, 1, :], dim=-1)
                    self.gripper_capture_pad_gap_m = float(pad_gap.max().detach().cpu().item())
            if self.gripper_probe_body_ids:
                robot = self.env.scene["robot"]
                probe_pos_w = robot.data.body_pos_w[:, self.gripper_probe_body_ids, :]
                self.gripper_capture_body_positions_w = {
                    name: pos
                    for name, pos in zip(
                        self.gripper_probe_body_names,
                        probe_pos_w[0].detach().cpu().tolist(),
                        strict=False,
                    )
                }
        captured_pad_offset = object_b - self.gripper_anchor_b(policy_obs)
        offset_error = captured_pad_offset - self.centered_grasp_offset_b
        self.gripper_capture_error_m = float(
            torch.linalg.vector_norm(offset_error, dim=-1).max().detach().cpu().item()
        )
        self.gripper_capture_offset_b = captured_pad_offset[0].detach().cpu().tolist()
        self.gripper_attach_snap_m = 0.0
        self.attached_pad_offset_b = captured_pad_offset.clone()

        eef_quat_b = policy_obs["eef_quat"].reshape(self.num_envs, 4).clone()
        eef_quat_b = eef_quat_b / torch.linalg.vector_norm(eef_quat_b, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.attached_offset_eef_b = math_utils.quat_apply_inverse(eef_quat_b, self.attached_offset_b)
        self.attached_offset_goal_b = self.attached_offset_static_b.clone()
        self.attached_offset_goal_b[:, 2] = -GRASP_EEF_Z_OFFSET
        self.attached_offset_locked = True
        self.carry_eef_b = self.eef_for_object(self.carry_object_b)
        self.carried_eef_b = eef_b.clone()

    def blend_attached_offset(self) -> None:
        if self.attached_offset_goal_b is None or not self.attached_offset_locked:
            return
        if self.phase_name in ("close",):
            return
        offset_b = self.attached_offset_static_b if self.attached_offset_static_b is not None else self.attached_offset_b
        z_delta = self.attached_offset_goal_b[:, 2:3] - offset_b[:, 2:3]
        z_step = torch.clamp(z_delta, min=-OFFSET_Z_BLEND_STEP_M, max=OFFSET_Z_BLEND_STEP_M)
        offset_b[:, 2:3] = offset_b[:, 2:3] + z_step
        if self.attached_offset_static_b is None:
            self.attached_offset_b = offset_b

    def clamp_object_into_gripper(self, policy_obs: dict[str, torch.Tensor], quat_w: torch.Tensor) -> None:
        target_w = self.centered_grasp_object_pos_w(policy_obs)
        target_w[:, 2] = self.source_object_pos_w[:, 2]
        current = self.object_cmd_pos_w.clone()
        delta = target_w - current
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        ratio = torch.clamp(PRECLAMP_OBJECT_STEP_M / (dist + 1.0e-6), max=1.0)
        new_pos = current + delta * ratio
        step = torch.linalg.vector_norm(new_pos - current, dim=-1).max().item()
        self.preclamp_max_object_step_m = max(self.preclamp_max_object_step_m, float(step))
        self.preclamp_final_error_m = float(
            torch.linalg.vector_norm(new_pos[:, :3] - target_w[:, :3], dim=-1).max().detach().cpu().item()
        )
        self.record_object_step(float(step), current, new_pos)
        self.object_cmd_pos_w = new_pos.clone()
        self.object_cmd_quat_w = quat_w.clone()
        set_rigid_pose(self.env, self.spec.base.object_name, self.object_cmd_pos_w, quat_w)

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
        self.motion_phase_count += 1
        pos, quat = self.nonholonomic_pose_for_segment(phase_index, self.motion_phase_count, segment_steps)
        set_robot_root_pose(self.env, pos, quat)
        if self.motion_phase_count >= segment_steps:
            self.advance_phase()

    def nonholonomic_pose_for_segment(
        self, phase_index: int, elapsed_steps: int, segment_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.route_waypoints[:, phase_index, :]
        end = self.route_waypoints[:, phase_index + 1, :]
        start_yaw = self.segment_start_yaws[:, phase_index]
        drive_yaw = self.segment_drive_yaws[:, phase_index]
        end_yaw = self.segment_end_yaws[:, phase_index]
        turn_in_steps = min(max(36, int(round(float(segment_steps) * 0.36))), max(segment_steps - 2, 1))
        turn_out_steps = 0
        if phase_index == len(self.base_motion_phases) - 1:
            final_turn = torch.abs(wrap_angle(end_yaw - drive_yaw)).max().item()
            if final_turn > 0.03:
                turn_out_steps = min(
                    max(36, int(round(float(segment_steps) * 0.30))), max(segment_steps - turn_in_steps - 1, 0)
                )
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

    def assist_object(self, policy_obs: dict[str, torch.Tensor]) -> None:
        quat = self.source_object_quat_w
        if self.attached:
            self.update_attached_offset(policy_obs)
            self.carried_eef_b = policy_obs["eef_pos"].reshape(self.num_envs, 3).clone()
            target_pos_w = self.attached_object_pos_w(policy_obs)
            current = self.object_cmd_pos_w.clone()
            self.object_cmd_pos_w = target_pos_w.clone()
            self.object_cmd_quat_w = quat.clone()
            step = torch.linalg.vector_norm(self.object_cmd_pos_w[:, :3] - current[:, :3], dim=-1).max().item()
            self.record_object_step(float(step), current, self.object_cmd_pos_w)
            set_rigid_pose(self.env, self.spec.base.object_name, self.object_cmd_pos_w, quat)
            sync_error = torch.linalg.vector_norm(self.object_cmd_pos_w[:, :3] - target_pos_w[:, :3], dim=-1)
            self.attached_sync_error_last_m = float(sync_error.max().detach().cpu().item())
            self.attached_sync_error_max_m = max(self.attached_sync_error_max_m, self.attached_sync_error_last_m)
            actual_object_b = world_to_base(self.env, self.object_cmd_pos_w)
            expected_object_b = world_to_base(self.env, target_pos_w)
            gripper_error = torch.linalg.vector_norm(actual_object_b - expected_object_b, dim=-1)
            self.gripper_object_sync_error_last_m = float(gripper_error.max().detach().cpu().item())
            self.gripper_object_sync_error_max_m = max(
                self.gripper_object_sync_error_max_m,
                self.gripper_object_sync_error_last_m,
            )
            gripper_anchor_b = self.gripper_anchor_b(policy_obs)
            object_body_clearance = torch.linalg.vector_norm(actual_object_b[:, :2], dim=-1) - BODY_KEEP_OUT_RADIUS_M
            gripper_body_clearance = torch.linalg.vector_norm(gripper_anchor_b[:, :2], dim=-1) - BODY_KEEP_OUT_RADIUS_M
            self.min_carried_object_body_clearance_m = min(
                self.min_carried_object_body_clearance_m,
                float(object_body_clearance.min().detach().cpu().item()),
            )
            self.min_carried_gripper_body_clearance_m = min(
                self.min_carried_gripper_body_clearance_m,
                float(gripper_body_clearance.min().detach().cpu().item()),
            )
        elif self.phase_name == "close":
            self.clamp_object_into_gripper(policy_obs, quat)
        elif self.released:
            self.release_settle_steps += 1
            target_w = torch.tensor(CUBE_TARGET_POS, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
            if float(torch.linalg.vector_norm(self.object_cmd_pos_w[:, :3] - target_w[:, :3], dim=-1).max().item()) > 0.006:
                self.move_object_toward(target_w, quat, 0.012)
            else:
                current = self.object_cmd_pos_w.clone()
                self.object_cmd_pos_w = target_w.clone()
                self.object_cmd_quat_w = quat.clone()
                step = torch.linalg.vector_norm(self.object_cmd_pos_w[:, :3] - current[:, :3], dim=-1).max().item()
                self.record_object_step(float(step), current, self.object_cmd_pos_w)
                set_rigid_pose(self.env, self.spec.base.object_name, self.object_cmd_pos_w, quat)
        else:
            displacement = torch.linalg.vector_norm(self.object_cmd_pos_w[:, :3] - self.source_object_pos_w[:, :3], dim=-1)
            self.pregrasp_max_object_motion_m = max(
                self.pregrasp_max_object_motion_m, float(displacement.max().detach().cpu().item())
            )
            self.object_cmd_pos_w = self.source_object_pos_w.clone()
            self.object_cmd_quat_w = quat.clone()
            set_rigid_pose(self.env, self.spec.base.object_name, self.source_object_pos_w, quat)

    def move_object_toward(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, max_step_m: float) -> None:
        current = self.object_cmd_pos_w.clone()
        delta = target_pos_w - current
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        ratio = torch.clamp(max_step_m / (dist + 1.0e-6), max=1.0)
        new_pos = current + delta * ratio
        step = torch.linalg.vector_norm(new_pos - current, dim=-1).max().item()
        self.record_object_step(float(step), current, new_pos)
        self.object_cmd_pos_w = new_pos.clone()
        self.object_cmd_quat_w = target_quat_w.clone()
        set_rigid_pose(self.env, self.spec.base.object_name, new_pos, target_quat_w)

    def record_object_step(self, step: float, from_w: torch.Tensor, to_w: torch.Tensor) -> None:
        if step <= self.object_max_step_m:
            return
        self.object_max_step_m = step
        self.object_max_step_phase = self.phase_name
        self.object_max_step_from_w = from_w[0].detach().cpu().tolist()
        self.object_max_step_to_w = to_w[0].detach().cpu().tolist()

    @property
    def phase_name(self) -> str:
        if self.phase_index >= len(self.phase_names):
            return "done"
        return self.phase_names[self.phase_index]

    @property
    def done(self) -> bool:
        return self.phase_index >= len(self.phase_names)


def main() -> None:
    args = parse_args()
    os.environ["USE_RELATIVE_MODE"] = "True"
    spec = make_spec()
    if args.record_frames:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_step_s = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(args.max_steps) * env_step_s * 1.3)
    configure_mobile_source_table(env_cfg, spec)
    add_mobile_target_table(env_cfg, spec)
    add_isaac_warehouse_scene(env_cfg, spec)
    configure_cube_object(env_cfg)
    if args.record_frames:
        add_mobile_demo_camera(env_cfg)

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    frames_dir = Path(args.frames_dir or "artifacts/frames/agibot_a2d_cube_pickplace_qhd")
    if args.record_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    source_pos = torch.tensor(CUBE_SOURCE_POS, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    source_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    source_quat[:, 0] = 1.0
    set_rigid_pose(env, "mug", source_pos, source_quat)
    with torch.inference_mode():
        for _ in range(12):
            obs, _, _, _, _ = env.step(hold_current_pose_action(env, obs["policy"], gripper=1.0))
            set_rigid_pose(env, "mug", source_pos, source_quat)

    table_aabbs_xy = get_table_aabbs_xy()
    static_obstacle_aabbs_xy = get_static_obstacle_aabbs_xy()
    all_obstacle_aabbs_xy = {**table_aabbs_xy, **static_obstacle_aabbs_xy}
    controller = AgibotA2DCubeController(spec, env, obs["policy"], args.drive_steps, table_aabbs_xy)
    min_static_obstacle_clearance_m, _ = route_clearance(
        controller.route_waypoints[0], all_obstacle_aabbs_xy, controller.robot_base_radius
    )
    static_obstacle_collision_check_passed = min_static_obstacle_clearance_m >= STATIC_OBSTACLE_CLEARANCE_THRESHOLD_M
    if args.record_frames:
        set_camera_view(env, env_step_s)

    frame_count = 0
    previous_frame_path: Path | None = None
    phase_log: list[dict[str, object]] = []
    last_phase = None
    gripper_events: list[dict[str, object]] = []
    last_gripper_sign = None
    gripper_transitions = 0
    min_runtime_table_clearance = float("inf")
    pick_descend_min_xy_error_m = float("inf")
    pick_descend_min_grasp_height_error_m = float("inf")
    release_cube_table_height_error_m = None
    done_step = None

    with torch.inference_mode():
        for step in range(args.max_steps):
            phase = controller.phase_name
            if phase != last_phase:
                phase_log.append({"step": step, "phase": phase})
                last_phase = phase
            controller.command_lift(step)
            action, _ = controller.act(obs["policy"])
            gripper_sign = 1 if float(action[0, -1].detach().cpu().item()) >= 0.0 else -1
            if last_gripper_sign is None or gripper_sign != last_gripper_sign:
                if last_gripper_sign is not None:
                    gripper_transitions += 1
                gripper_events.append({"step": step, "phase": phase, "command": gripper_sign})
            last_gripper_sign = gripper_sign

            obs, _, _, _, _ = env.step(action)
            controller.command_gripper_aperture()
            controller.after_step()
            controller.command_gripper_aperture()
            controller.assist_object(obs["policy"])
            controller.tick_phase()

            object_b = world_to_base(env, controller.object_cmd_pos_w)
            eef_b = obs["policy"]["eef_pos"].reshape(env.num_envs, 3)
            if phase in ("descend", "close"):
                gripper_anchor_b = controller.gripper_anchor_b(obs["policy"])
                target_anchor_b = object_b - controller.centered_grasp_offset_b
                xy_error = torch.linalg.vector_norm(gripper_anchor_b[:, :2] - target_anchor_b[:, :2], dim=-1)
                grasp_height_error = torch.abs(gripper_anchor_b[:, 2] - target_anchor_b[:, 2])
                pick_descend_min_xy_error_m = min(
                    pick_descend_min_xy_error_m, float(xy_error.max().detach().cpu().item())
                )
                pick_descend_min_grasp_height_error_m = min(
                    pick_descend_min_grasp_height_error_m,
                    float(grasp_height_error.max().detach().cpu().item()),
                )
            if controller.released:
                release_error = height_error_on_table(controller.object_cmd_pos_w, TARGET_TABLE_POS[2])
                release_cube_table_height_error_m = (
                    release_error
                    if release_cube_table_height_error_m is None
                    else min(release_cube_table_height_error_m, release_error)
                )

            base_xy = env.scene["robot"].data.root_pos_w[:, :2]
            min_runtime_table_clearance = min(
                min_runtime_table_clearance,
                table_penetration_clearance(base_xy, table_aabbs_xy, controller.robot_base_radius),
            )

            if args.record_frames and step % max(args.capture_every, 1) == 0:
                frame_path = frames_dir / f"frame_{frame_count:04d}.png"
                frame_ok = record_camera_frame(env, frame_path, env_step_s, previous_frame_path)
                if not frame_ok:
                    print(f"WARNING: recorded frame appears black: {frame_path}", flush=True)
                else:
                    previous_frame_path = frame_path
                frame_count += 1

            target_w = torch.tensor(CUBE_TARGET_POS, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
            command_error = float(torch.linalg.vector_norm(controller.object_cmd_pos_w - target_w, dim=-1).max().item())
            if controller.released and controller.release_settle_steps >= POST_RELEASE_SETTLE_STEPS and command_error < 0.004:
                done_step = step
                break
            if controller.done:
                done_step = step
                break

    robot = env.scene["robot"]
    object_pos = env.scene["mug"].data.root_pos_w[0].detach().cpu().tolist()
    object_quat = env.scene["mug"].data.root_quat_w[0].detach().cpu().tolist()
    final_target = torch.tensor(CUBE_TARGET_POS, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    object_target_error_m = float(
        torch.linalg.vector_norm(env.scene["mug"].data.root_pos_w[:, :3] - final_target, dim=-1).max().item()
    )
    command_target_error_m = float(
        torch.linalg.vector_norm(controller.object_cmd_pos_w[:, :3] - final_target, dim=-1).max().item()
    )
    final_base = robot.data.root_pos_w[0].detach().cpu().tolist()
    base_goal_error_m = float(
        torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - controller.root_goal_pos[:, :2], dim=-1).max().item()
    )
    lift_range = 0.0 if controller.lift_min == float("inf") else controller.lift_max - controller.lift_min
    source_cube_table_height_error_m = height_error_on_table(source_pos, SOURCE_TABLE_POS[2])
    final_cube_table_height_error_m = height_error_on_table(env.scene["mug"].data.root_pos_w[:, :3], TARGET_TABLE_POS[2])
    pick_descend_reached = (
        pick_descend_min_xy_error_m <= PICK_XY_TOLERANCE_M
        and pick_descend_min_grasp_height_error_m <= CUBE_TABLE_HEIGHT_TOLERANCE_M * 2.0
    )
    release_cube_table_height_error_m = (
        final_cube_table_height_error_m
        if release_cube_table_height_error_m is None
        else release_cube_table_height_error_m
    )
    success = bool(
        command_target_error_m < 0.006
        and object_target_error_m < 0.008
        and controller.collision_check_passed
        and min_runtime_table_clearance >= TABLE_CLEARANCE_THRESHOLD_M
        and static_obstacle_collision_check_passed
        and lift_range > 0.06
        and source_cube_table_height_error_m <= CUBE_TABLE_HEIGHT_TOLERANCE_M
        and final_cube_table_height_error_m <= CUBE_TABLE_HEIGHT_TOLERANCE_M
        and release_cube_table_height_error_m <= CUBE_TABLE_HEIGHT_TOLERANCE_M
        and pick_descend_reached
        and controller.attached_offset_locked
        and controller.gripper_capture_error_m <= GRIPPER_CAPTURE_ERROR_THRESHOLD_M
        and controller.attached_sync_error_max_m <= ATTACHED_SYNC_ERROR_THRESHOLD_M
        and controller.gripper_object_sync_error_max_m <= ATTACHED_SYNC_ERROR_THRESHOLD_M
        and controller.min_carried_object_body_clearance_m >= 0.0
        and controller.object_max_step_m <= OBJECT_TELEPORT_STEP_THRESHOLD_M
    )
    curve_samples = controller.route_samples_xy
    curve_deltas = curve_samples[1:] - curve_samples[:-1] if curve_samples.shape[0] > 1 else curve_samples
    base_route_length = (
        float(torch.linalg.vector_norm(curve_deltas, dim=-1).sum().item()) if curve_samples.shape[0] > 1 else 0.0
    )
    summary = {
        "task": TASK,
        "robot": "Agibot A2D",
        "visual_object": "single_scaled_green_cube",
        "success": success,
        "done_step": done_step,
        "final_phase": controller.phase_name,
        "object_name_in_env": "mug",
        "object_pos_w": object_pos,
        "object_quat_w": object_quat,
        "object_target_pos_w": list(CUBE_TARGET_POS),
        "object_target_error_m": object_target_error_m,
        "object_command_target_error_m": command_target_error_m,
        "object_max_step_m": controller.object_max_step_m,
        "object_max_step_phase": controller.object_max_step_phase,
        "object_max_step_from_w": controller.object_max_step_from_w,
        "object_max_step_to_w": controller.object_max_step_to_w,
        "object_teleport_step_threshold_m": OBJECT_TELEPORT_STEP_THRESHOLD_M,
        "pregrasp_max_object_motion_m": controller.pregrasp_max_object_motion_m,
        "lift_driven_pick_phases": sorted(LIFT_DRIVEN_PICK_PHASES),
        "lift_driven_place_phases": sorted(LIFT_DRIVEN_PLACE_PHASES),
        "object_attachment_mode": "left_gripper_pad_midpoint_root_locked_partial_gripper",
        "gripper_capture_error_m": controller.gripper_capture_error_m,
        "gripper_capture_offset_b": controller.gripper_capture_offset_b,
        "gripper_capture_anchor_w": controller.gripper_capture_anchor_w,
        "gripper_capture_anchor_quat_w": controller.gripper_capture_anchor_quat_w,
        "gripper_capture_center_w": controller.gripper_capture_center_w,
        "gripper_capture_pad_positions_w": controller.gripper_capture_pad_positions_w,
        "gripper_capture_pad_gap_m": controller.gripper_capture_pad_gap_m,
        "gripper_capture_body_positions_w": controller.gripper_capture_body_positions_w,
        "gripper_centered_grasp_offset_b": list(GRIPPER_CENTERED_GRASP_OFFSET_B),
        "gripper_hold_joint_position": GRIPPER_CUBE_HOLD_JOINT_POS,
        "gripper_hold_joint_names": controller.gripper_hold_joint_names,
        "gripper_pad_gap_target_m": CUBE_HALF_HEIGHT * 2.0,
        "gripper_capture_error_threshold_m": GRIPPER_CAPTURE_ERROR_THRESHOLD_M,
        "gripper_attach_snap_m": controller.gripper_attach_snap_m,
        "preclamp_object_step_m": PRECLAMP_OBJECT_STEP_M,
        "preclamp_max_object_step_m": controller.preclamp_max_object_step_m,
        "preclamp_final_error_m": controller.preclamp_final_error_m,
        "attached_sync_error_max_m": controller.attached_sync_error_max_m,
        "attached_sync_error_last_m": controller.attached_sync_error_last_m,
        "attached_sync_error_threshold_m": ATTACHED_SYNC_ERROR_THRESHOLD_M,
        "gripper_object_sync_error_max_m": controller.gripper_object_sync_error_max_m,
        "gripper_object_sync_error_last_m": controller.gripper_object_sync_error_last_m,
        "min_carried_object_body_clearance_m": controller.min_carried_object_body_clearance_m,
        "min_carried_gripper_body_clearance_m": controller.min_carried_gripper_body_clearance_m,
        "body_keep_out_radius_m": BODY_KEEP_OUT_RADIUS_M,
        "pick_descend_reached": pick_descend_reached,
        "pick_descend_min_xy_error_m": pick_descend_min_xy_error_m,
        "pick_descend_min_grasp_height_error_m": pick_descend_min_grasp_height_error_m,
        "cube_half_height_m": CUBE_HALF_HEIGHT,
        "cube_starts_on_table": abs(CUBE_SOURCE_POS[2] - (SOURCE_TABLE_POS[2] + CUBE_HALF_HEIGHT)) < 1.0e-6,
        "cube_ends_on_table": abs(CUBE_TARGET_POS[2] - (TARGET_TABLE_POS[2] + CUBE_HALF_HEIGHT)) < 1.0e-6,
        "source_cube_table_height_error_m": source_cube_table_height_error_m,
        "release_cube_table_height_error_m": release_cube_table_height_error_m,
        "final_cube_table_height_error_m": final_cube_table_height_error_m,
        "cube_table_height_tolerance_m": CUBE_TABLE_HEIGHT_TOLERANCE_M,
        "grasp_eef_z_offset_m": GRASP_EEF_Z_OFFSET,
        "attached_offset_b": controller.attached_offset_b[0].detach().cpu().tolist(),
        "attached_pad_offset_b": (
            None
            if controller.attached_pad_offset_b is None
            else controller.attached_pad_offset_b[0].detach().cpu().tolist()
        ),
        "gripper_center_body_names": controller.gripper_center_body_names,
        "left_pad_body_names": controller.left_pad_body_names,
        "attached_offset_locked": controller.attached_offset_locked,
        "release_pose_w": controller.release_pose_w,
        "post_release_settle_steps": controller.release_settle_steps,
        "post_release_settle_steps_required": POST_RELEASE_SETTLE_STEPS,
        "lift_joint_name": "joint_lift_body",
        "lift_command_mode": "scripted_smooth_joint_position_with_arm_rmpflow",
        "lift_joint_range": [controller.lift_min, controller.lift_max],
        "lift_range_m": lift_range,
        "lift_trace": controller.lift_trace,
        "gripper_command_events": gripper_events,
        "gripper_command_transitions": gripper_transitions,
        "base_motion_mode": "nonholonomic_turn_drive_turn_root_motion",
        "base_start_pos_w": controller.root_start_pos[0].detach().cpu().tolist(),
        "base_goal_pos_w": controller.root_goal_pos[0].detach().cpu().tolist(),
        "base_final_pos_w": final_base,
        "base_goal_error_m": base_goal_error_m,
        "base_route_length_m": base_route_length,
        "route_waypoints_w": controller.route_waypoints[0].detach().cpu().tolist(),
        "base_segment_drive_yaws_rad": controller.segment_drive_yaws[0].detach().cpu().tolist(),
        "collision_check_passed": controller.collision_check_passed,
        "collision_clearance_threshold_m": TABLE_CLEARANCE_THRESHOLD_M,
        "min_table_clearance_m": controller.min_table_clearance_m,
        "min_runtime_table_clearance_m": min_runtime_table_clearance,
        "static_obstacle_collision_check_passed": static_obstacle_collision_check_passed,
        "static_obstacle_clearance_threshold_m": STATIC_OBSTACLE_CLEARANCE_THRESHOLD_M,
        "min_static_obstacle_clearance_m": min_static_obstacle_clearance_m,
        "robot_base_radius_m": controller.robot_base_radius,
        "table_aabbs_xy": table_aabbs_xy,
        "static_obstacle_aabbs_xy": static_obstacle_aabbs_xy,
        "source_table_pos_w": list(SOURCE_TABLE_POS),
        "target_table_pos_w": list(TARGET_TABLE_POS),
        "table_distance_m": math.dist(SOURCE_TABLE_POS[:2], TARGET_TABLE_POS[:2]),
        "final_eef_pos_b": obs["policy"]["eef_pos"][0].detach().cpu().tolist(),
        "final_eef_pos_w": base_to_world(env, obs["policy"]["eef_pos"].reshape(env.num_envs, 3))[0]
        .detach()
        .cpu()
        .tolist(),
        "initial_eef_pos_b": controller.initial_eef_b,
        "phase_log": phase_log,
        "scene_style": "isaac_simple_warehouse_enlarged",
        "camera_resolution": [QHD_WIDTH, QHD_HEIGHT] if args.record_frames else None,
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
