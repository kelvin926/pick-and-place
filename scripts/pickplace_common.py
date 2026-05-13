from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TaskSpec:
    task: str
    label: str
    object_name: str
    object_pos_key: str
    target_world_pos: tuple[float, float, float]
    target_table_pos: tuple[float, float, float]
    target_table_scale: tuple[float, float, float]
    grasp_z: float
    approach_z: float
    lift_z: float
    place_z: float
    hold_offset: tuple[float, float, float]
    carry_quat_world: tuple[float, float, float, float]
    secondary_name: str | None = None
    secondary_world_pos: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class MobileTaskSpec:
    base: TaskSpec
    source_table_pos: tuple[float, float, float]
    target_world_pos: tuple[float, float, float]
    target_table_pos: tuple[float, float, float]
    target_table_scale: tuple[float, float, float]
    secondary_world_pos: tuple[float, float, float] | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0": TaskSpec(
        task="Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0",
        label="galbot_cube",
        object_name="cube_3",
        object_pos_key="cube_positions",
        target_world_pos=(0.12, 0.55, 0.0203),
        target_table_pos=(0.18, 0.58, 0.0),
        target_table_scale=(0.28, 0.28, 0.32),
        grasp_z=0.035,
        approach_z=0.18,
        lift_z=0.24,
        place_z=0.035,
        hold_offset=(0.0, 0.0, -0.035),
        carry_quat_world=(1.0, 0.0, 0.0, 0.0),
    ),
    "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0": TaskSpec(
        task="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
        label="agibot_mug",
        object_name="mug",
        object_pos_key="mug_positions",
        target_world_pos=(0.24, -0.36, 0.76),
        target_table_pos=(0.30, -0.46, 0.60),
        target_table_scale=(0.75, 0.75, 0.60),
        grasp_z=0.05,
        approach_z=0.22,
        lift_z=0.30,
        place_z=0.08,
        hold_offset=(0.0, 0.0, -0.055),
        carry_quat_world=(1.0, 0.0, 0.0, 0.0),
    ),
    "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0": TaskSpec(
        task="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
        label="agibot_toy2box",
        object_name="toy_truck",
        object_pos_key="toy_truck_positions",
        target_world_pos=(0.30, 0.34, -0.49),
        target_table_pos=(0.50, 0.50, -0.70),
        target_table_scale=(1.00, 0.70, 0.30),
        grasp_z=0.055,
        approach_z=0.22,
        lift_z=0.30,
        place_z=0.08,
        hold_offset=(0.0, 0.0, -0.060),
        carry_quat_world=(1.0, 0.0, 0.0, 0.0),
        secondary_name="box",
        secondary_world_pos=(0.30, 0.34, -0.55),
    ),
}


MOBILE_TASK_SPECS: dict[str, MobileTaskSpec] = {
    "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0": MobileTaskSpec(
        base=TASK_SPECS["Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0"],
        source_table_pos=(1.10, 0.0, 0.0),
        target_world_pos=(4.74, -0.35, 0.0203),
        target_table_pos=(4.80, 0.0, 0.0),
        target_table_scale=(1.0, 1.0, 1.0),
    ),
    "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0": MobileTaskSpec(
        base=TASK_SPECS["Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0"],
        source_table_pos=(1.10, 0.0, 0.60),
        target_world_pos=(4.54, -0.36, 0.76),
        target_table_pos=(4.80, 0.0, 0.60),
        target_table_scale=(1.0, 1.0, 0.60),
    ),
    "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0": MobileTaskSpec(
        base=TASK_SPECS["Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"],
        source_table_pos=(1.10, 0.0, -0.70),
        target_world_pos=(4.60, -0.70, -0.49),
        target_table_pos=(4.80, 0.0, -0.70),
        target_table_scale=(1.8, 1.0, 0.30),
        secondary_world_pos=(4.60, -0.70, -0.55),
    ),
}


OBS_KEY_ORDER: dict[str, list[str]] = {
    "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0": [
        "actions",
        "joint_pos",
        "joint_vel",
        "object",
        "cube_positions",
        "cube_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ],
    "Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0": [
        "actions",
        "joint_pos",
        "joint_vel",
        "mug_positions",
        "mug_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ],
    "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0": [
        "actions",
        "joint_pos",
        "joint_vel",
        "toy_truck_positions",
        "toy_truck_orientations",
        "box_positions",
        "box_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ],
}


def get_spec(task: str) -> TaskSpec:
    task = task.split(":")[-1]
    if task not in TASK_SPECS:
        raise KeyError(f"Unsupported task: {task}")
    return TASK_SPECS[task]


def get_mobile_spec(task: str) -> MobileTaskSpec:
    task = task.split(":")[-1]
    if task not in MOBILE_TASK_SPECS:
        raise KeyError(f"Unsupported mobile task: {task}")
    return MOBILE_TASK_SPECS[task]


def add_second_table(env_cfg: Any, spec: TaskSpec) -> None:
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    env_cfg.scene.target_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(spec.target_table_pos),
            rot=[0.707, 0.0, 0.0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=spec.target_table_scale,
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
        ),
    )


def add_mobile_target_table(env_cfg: Any, spec: MobileTaskSpec) -> None:
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    env_cfg.scene.target_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(spec.target_table_pos),
            rot=[0.707, 0.0, 0.0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=spec.target_table_scale,
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


def configure_mobile_source_table(env_cfg: Any, spec: MobileTaskSpec) -> None:
    from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg

    table_cfg = getattr(env_cfg.scene, "table", None)
    if table_cfg is None:
        return
    table_cfg.init_state.pos = list(spec.source_table_pos)
    if getattr(table_cfg, "spawn", None) is not None:
        table_cfg.spawn.scale = spec.target_table_scale
        table_cfg.spawn.collision_props = CollisionPropertiesCfg(collision_enabled=True)


def add_industrial_scene(env_cfg: Any, spec: MobileTaskSpec) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    floor_offsets = {
        "galbot_cube": -0.82,
        "agibot_mug": -0.62,
        "agibot_toy2box": -0.37,
    }
    floor_z = spec.source_table_pos[2] + floor_offsets.get(spec.base.label, -0.62)
    no_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

    def mat(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.85)

    def cuboid(
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        setattr(
            env_cfg.scene,
            f"decor_{name}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/IndustrialDecor_{name}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=no_collision,
                    visual_material=mat(color),
                ),
            ),
        )

    env_cfg.scene.decor_dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/IndustrialDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2800.0, color=(0.78, 0.82, 0.86)),
    )
    env_cfg.scene.decor_key_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/IndustrialKeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, -0.3827, 0.0, 0.0)),
        spawn=sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 0.96, 0.88)),
    )

    # Factory floor and AMR safety lane.
    cuboid("concrete_floor", (1.75, -0.10, floor_z - 0.015), (6.4, 4.8, 0.03), (0.24, 0.25, 0.25))
    for index, y in enumerate((-2.15, -0.55)):
        cuboid(f"yellow_lane_line_{index}", (1.75, y, floor_z + 0.006), (5.8, 0.045, 0.012), (1.0, 0.72, 0.08))
    for index, x in enumerate((-0.40, 3.90)):
        cuboid(f"docking_zone_{index}_front", (x, -0.40, floor_z + 0.008), (0.95, 0.045, 0.012), (1.0, 0.72, 0.08))
        cuboid(f"docking_zone_{index}_back", (x, 0.62, floor_z + 0.008), (0.95, 0.045, 0.012), (1.0, 0.72, 0.08))
        cuboid(f"docking_zone_{index}_left", (x - 0.48, 0.11, floor_z + 0.008), (0.045, 1.02, 0.012), (1.0, 0.72, 0.08))
        cuboid(f"docking_zone_{index}_right", (x + 0.48, 0.11, floor_z + 0.008), (0.045, 1.02, 0.012), (1.0, 0.72, 0.08))

    # Background storage rack, crates, and conveyor-like work area. These are visual only and kept outside the route.
    rack_y = 1.95
    for index, x in enumerate((-0.55, 0.95, 2.45, 3.95)):
        cuboid(f"rack_post_{index}", (x, rack_y, floor_z + 0.95), (0.06, 0.08, 1.90), (0.08, 0.09, 0.10))
    for index, z in enumerate((0.46, 0.98, 1.50)):
        cuboid(f"rack_beam_{index}", (1.70, rack_y, floor_z + z), (4.60, 0.10, 0.07), (0.95, 0.50, 0.08))
    crate_colors = ((0.12, 0.32, 0.70), (0.72, 0.16, 0.12), (0.18, 0.55, 0.32), (0.72, 0.55, 0.16))
    crate_id = 0
    for shelf_z in (0.64, 1.16, 1.68):
        for x in (-0.05, 1.10, 2.25, 3.35):
            cuboid(
                f"rack_crate_{crate_id}",
                (x, rack_y - 0.08, floor_z + shelf_z),
                (0.42, 0.42, 0.28),
                crate_colors[crate_id % len(crate_colors)],
            )
            crate_id += 1

    for index, x in enumerate((4.55, 4.90)):
        cuboid(f"pallet_stack_{index}", (x, -2.35, floor_z + 0.17), (0.62, 0.52, 0.34), (0.50, 0.34, 0.16))
    for index, x in enumerate((4.55, 4.90)):
        cuboid(f"carton_stack_{index}", (x, -2.35, floor_z + 0.50), (0.48, 0.42, 0.30), (0.67, 0.47, 0.25))

    for index, x in enumerate((-0.80, 4.15)):
        cuboid(f"warning_bollard_{index}", (x, -0.86, floor_z + 0.34), (0.10, 0.10, 0.68), (1.0, 0.52, 0.04))
        cuboid(f"warning_bollard_cap_{index}", (x, -0.86, floor_z + 0.70), (0.16, 0.16, 0.06), (0.06, 0.06, 0.06))

    cuboid("overhead_light_left", (0.20, 0.25, floor_z + 2.35), (1.35, 0.08, 0.05), (0.95, 0.95, 0.86))
    cuboid("overhead_light_right", (3.15, 0.25, floor_z + 2.35), (1.35, 0.08, 0.05), (0.95, 0.95, 0.86))
    cuboid("rear_wall", (1.75, 2.42, floor_z + 1.05), (6.7, 0.08, 2.15), (0.34, 0.36, 0.36))
    cuboid("left_column", (-1.35, 1.15, floor_z + 1.10), (0.16, 0.16, 2.20), (0.18, 0.20, 0.21))
    cuboid("right_column", (4.85, 1.15, floor_z + 1.10), (0.16, 0.16, 2.20), (0.18, 0.20, 0.21))

    for index, x in enumerate((-0.35, 0.65, 1.65, 2.65, 3.65)):
        cuboid(f"safety_fence_post_{index}", (x, 1.28, floor_z + 0.52), (0.055, 0.055, 1.04), (0.04, 0.05, 0.05))
    for index, z in enumerate((0.32, 0.72)):
        cuboid(f"safety_fence_rail_{index}", (1.65, 1.28, floor_z + z), (4.10, 0.045, 0.055), (0.96, 0.66, 0.06))

    cuboid("conveyor_frame", (1.65, 1.05, floor_z + 0.36), (2.05, 0.45, 0.12), (0.12, 0.13, 0.14))
    cuboid("conveyor_belt", (1.65, 1.05, floor_z + 0.445), (1.90, 0.36, 0.035), (0.03, 0.035, 0.04))
    for index, x in enumerate((0.90, 1.25, 1.60, 1.95, 2.30)):
        cuboid(f"pcb_tray_{index}", (x, 1.05, floor_z + 0.50), (0.24, 0.18, 0.025), (0.08, 0.48, 0.24))

    cuboid("inspection_bench", (-0.85, 1.00, floor_z + 0.42), (0.70, 0.55, 0.10), (0.10, 0.10, 0.11))
    cuboid("inspection_monitor", (-0.85, 1.20, floor_z + 0.82), (0.36, 0.035, 0.25), (0.02, 0.03, 0.035))
    cuboid("inspection_monitor_glow", (-0.85, 1.175, floor_z + 0.82), (0.31, 0.012, 0.20), (0.08, 0.45, 0.75))


def add_isaac_warehouse_scene(env_cfg: Any, spec: MobileTaskSpec) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    floor_offsets = {
        "galbot_cube": -0.82,
        "agibot_mug": -0.62,
        "agibot_toy2box": -0.37,
    }
    floor_z = spec.source_table_pos[2] + floor_offsets.get(spec.base.label, -0.62)
    no_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

    def mat(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.82, metallic=0.0)

    def cuboid(
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        setattr(
            env_cfg.scene,
            f"warehouse_{name}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/WarehouseTrim_{name}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=no_collision,
                    visual_material=mat(color),
                ),
            ),
        )

    def usd_asset(
        name: str,
        usd_path: str,
        pos: tuple[float, float, float],
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        setattr(
            env_cfg.scene,
            f"warehouse_usd_{name}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/WarehouseAsset_{name}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=scale,
                    collision_props=no_collision,
                ),
            ),
        )

    env_cfg.scene.warehouse_dome_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WarehouseDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3600.0, color=(0.82, 0.86, 0.90)),
    )
    env_cfg.scene.warehouse_key_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WarehouseKeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, -0.3827, 0.0, 0.0)),
        spawn=sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.96, 0.90)),
    )

    # Large enclosing floor/walls keep the camera inside the warehouse instead of seeing the outside grid.
    cuboid("warehouse_floor", (2.75, -1.75, floor_z - 0.030), (10.60, 11.20, 0.035), (0.27, 0.28, 0.28))
    cuboid("workcell_floor_mat", (2.85, -0.42, floor_z - 0.010), (7.30, 4.80, 0.018), (0.18, 0.19, 0.19))
    cuboid("amr_lane_left", (2.70, -2.60, floor_z + 0.012), (7.40, 0.040, 0.010), (0.90, 0.66, 0.16))
    cuboid("amr_lane_right", (2.70, -0.82, floor_z + 0.012), (7.40, 0.040, 0.010), (0.90, 0.66, 0.16))
    cuboid("rear_wall", (2.75, 3.75, floor_z + 2.75), (10.60, 0.12, 5.50), (0.35, 0.36, 0.36))
    cuboid("left_wall", (-2.60, -0.95, floor_z + 2.75), (0.12, 9.40, 5.50), (0.30, 0.31, 0.32))
    cuboid("right_wall", (8.10, -0.95, floor_z + 2.75), (0.12, 9.40, 5.50), (0.30, 0.31, 0.32))
    cuboid("left_column", (-1.20, 2.20, floor_z + 1.90), (0.24, 0.24, 3.80), (0.16, 0.17, 0.18))
    cuboid("right_column", (6.70, 2.20, floor_z + 1.90), (0.24, 0.24, 3.80), (0.16, 0.17, 0.18))

    rack = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd"
    rack_pile = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd"
    rack_shelf = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd"
    box = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd"
    pallet = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd"
    crate = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd"
    forklift = f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd"

    rack_scale = (0.95, 0.95, 0.95)
    for index, x in enumerate((-0.85, 1.05, 2.95, 4.85, 6.75)):
        usd_asset(f"rack_frame_{index}", rack, (x, 2.30, floor_z), scale=rack_scale)
    for index, x in enumerate((0.10, 2.00, 3.90, 5.80)):
        usd_asset(f"rack_pile_{index}", rack_pile, (x, 2.34, floor_z), scale=rack_scale)
    for index, x in enumerate((0.10, 2.00, 3.90, 5.80)):
        usd_asset(f"rack_shelf_{index}", rack_shelf, (x, 2.34, floor_z + 1.12), scale=rack_scale)

    for index, (x, y, z) in enumerate(
        (
            (6.20, -3.35, 0.0),
            (6.90, -3.35, 0.0),
            (6.20, -2.78, 0.0),
            (6.90, -2.78, 0.0),
        )
    ):
        usd_asset(f"pallet_{index}", pallet, (x, y, floor_z + z), scale=(0.90, 0.90, 0.90))
        usd_asset(f"box_{index}", box, (x, y, floor_z + z + 0.38), scale=(0.90, 0.90, 0.90))

    for index, x in enumerate((5.65, 6.20, 6.75)):
        usd_asset(f"crate_{index}", crate, (x, 1.20, floor_z + 0.16), scale=(0.82, 0.82, 0.82))
    usd_asset("forklift", forklift, (6.95, 0.60, floor_z), rot=(0.0, 0.0, 0.0, 1.0), scale=(1.00, 1.00, 1.00))


def add_cube_frame_markers(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    no_collision = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

    def mat(color: tuple[float, float, float]) -> sim_utils.PreviewSurfaceCfg:
        return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.55)

    marker_specs = {
        "x": ((0.18, 0.014, 0.014), (0.95, 0.05, 0.04)),
        "y": ((0.014, 0.18, 0.014), (0.05, 0.80, 0.08)),
        "z": ((0.014, 0.014, 0.18), (0.06, 0.18, 0.95)),
    }
    for axis, (size, color) in marker_specs.items():
        setattr(
            env_cfg.scene,
            f"cube_frame_{axis}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/CubeFrame_{axis.upper()}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=no_collision,
                    visual_material=mat(color),
                ),
            ),
        )


def add_demo_camera(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        height=1440,
        width=2560,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.10, -1.05, 1.05),
            rot=(-0.3536, 0.6124, 0.6124, -0.3536),
            convention="ros",
        ),
    )


def add_mobile_demo_camera(env_cfg: Any) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.demo_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DemoCamera",
        update_period=0.0,
        height=1440,
        width=2560,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=500.0,
            horizontal_aperture=24.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.05, -4.60, 2.15),
            rot=(-0.3536, 0.6124, 0.6124, -0.3536),
            convention="ros",
        ),
    )


def flatten_policy_obs(policy_obs: dict[str, torch.Tensor], task: str) -> torch.Tensor:
    parts = []
    for key in OBS_KEY_ORDER[task]:
        value = policy_obs[key]
        parts.append(value.reshape(value.shape[0], -1))
    return torch.cat(parts, dim=-1)


def object_pos_from_obs(policy_obs: dict[str, torch.Tensor], spec: TaskSpec) -> torch.Tensor:
    value = policy_obs[spec.object_pos_key].reshape(policy_obs[spec.object_pos_key].shape[0], -1)
    if spec.object_pos_key == "cube_positions":
        if spec.object_name == "cube_2":
            return value[:, 3:6]
        if spec.object_name == "cube_3":
            return value[:, 6:9]
        return value[:, 0:3]
    return value[:, :3]


def base_to_world(env: Any, pos_b: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    robot = env.scene["robot"]
    quat_b = torch.zeros((pos_b.shape[0], 4), device=pos_b.device)
    quat_b[:, 0] = 1.0
    pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_b, quat_b)
    return pos_w


def world_to_base(env: Any, pos_w: torch.Tensor) -> torch.Tensor:
    import isaaclab.utils.math as math_utils

    robot = env.scene["robot"]
    quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device)
    quat_w[:, 0] = 1.0
    pos_b, _ = math_utils.subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, pos_w, quat_w)
    return pos_b


def set_rigid_pose(env: Any, asset_name: str, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None) -> None:
    asset = env.scene[asset_name]
    if quat_w is None:
        quat_w = torch.zeros((pos_w.shape[0], 4), device=pos_w.device)
        quat_w[:, 0] = 1.0
    root_pose = torch.cat([pos_w, quat_w], dim=-1)
    with torch.inference_mode():
        asset.write_root_pose_to_sim(root_pose)
        asset.write_root_velocity_to_sim(torch.zeros((pos_w.shape[0], 6), device=pos_w.device))


def set_secondary_target(env: Any, spec: TaskSpec) -> None:
    if spec.secondary_name is None or spec.secondary_world_pos is None:
        return
    pos_w = torch.tensor(spec.secondary_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    set_rigid_pose(env, spec.secondary_name, pos_w)


def set_mobile_secondary_target(env: Any, spec: MobileTaskSpec) -> None:
    if spec.base.secondary_name is None or spec.secondary_world_pos is None:
        return
    pos_w = torch.tensor(spec.secondary_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    set_rigid_pose(env, spec.base.secondary_name, pos_w)


def set_robot_root_pose(env: Any, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None) -> None:
    robot = env.scene["robot"]
    if quat_w is None:
        quat_w = robot.data.root_quat_w.clone()
    root_pose = torch.cat([pos_w, quat_w], dim=-1)
    with torch.inference_mode():
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((pos_w.shape[0], 6), device=pos_w.device))


def hold_current_pose_action(env: Any, policy_obs: dict[str, torch.Tensor], gripper: float = 1.0) -> torch.Tensor:
    action = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
    if action.shape[-1] == 8:
        action[:, :3] = base_to_world(env, policy_obs["eef_pos"].reshape(env.num_envs, 3))
        action[:, 3:7] = policy_obs["eef_quat"].reshape(env.num_envs, 4)
        action[:, 7] = gripper
    else:
        action[:, 6] = gripper
    return action


class ScriptedPickPlaceExpert:
    def __init__(self, spec: TaskSpec, env: Any, initial_policy_obs: dict[str, torch.Tensor]):
        self.spec = spec
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.action_dim = env.action_space.shape[-1]
        self.phase_index = 0
        self.hold_count = 0
        self.attached = False
        self.released = False
        self.object_start_b = object_pos_from_obs(initial_policy_obs, spec).clone()
        target_w = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        self.target_b = world_to_base(env, target_w)

    def _phase_target(self) -> tuple[torch.Tensor, float, int, bool, bool]:
        open_cmd = 1.0
        close_cmd = -1.0
        phases = [
            (self.object_start_b + self._z(self.spec.approach_z), open_cmd, 0, False, False),
            (self.object_start_b + self._z(self.spec.grasp_z), open_cmd, 0, False, False),
            (self.object_start_b + self._z(self.spec.grasp_z), close_cmd, 18, True, False),
            (self.object_start_b + self._z(self.spec.lift_z), close_cmd, 0, True, False),
            (self.target_b + self._z(self.spec.lift_z), close_cmd, 0, True, False),
            (self.target_b + self._z(self.spec.place_z), close_cmd, 0, True, False),
            (self.target_b + self._z(self.spec.place_z), open_cmd, 18, False, True),
            (self.target_b + self._z(self.spec.lift_z), open_cmd, 0, False, True),
        ]
        return phases[min(self.phase_index, len(phases) - 1)]

    def _z(self, value: float) -> torch.Tensor:
        out = torch.zeros((self.num_envs, 3), device=self.device)
        out[:, 2] = value
        return out

    def act(self, policy_obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
        target, gripper, hold_steps, attach, release = self._phase_target()
        eef = policy_obs["eef_pos"].reshape(self.num_envs, 3)
        delta = target - eef
        dist = torch.linalg.vector_norm(delta, dim=-1)

        action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        if self.action_dim == 8:
            action[:, :3] = base_to_world(self.env, target)
            action[:, 3:7] = policy_obs["eef_quat"].reshape(self.num_envs, 4)
            action[:, 7] = gripper
        else:
            max_delta = 0.006 if self.spec.label == "galbot_cube" else 0.035
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
                self.phase_index += 1
                self.hold_count = 0
        else:
            tolerance = 0.025
            if self.spec.label == "galbot_cube":
                tolerance = 0.075 if self.phase_index in (3, 4, 5, 7) else 0.035
            elif self.phase_index in (4, 5, 7):
                tolerance = 0.04
            if bool((dist < tolerance).all().item()):
                self.phase_index += 1
                self.hold_count = 0

        return action, self.phase_name

    @property
    def phase_name(self) -> str:
        names = ["approach", "descend", "close", "lift", "transfer", "place", "open", "retreat"]
        return names[min(self.phase_index, len(names) - 1)]

    @property
    def done(self) -> bool:
        return self.phase_index >= 8

    def assist_object(self, env: Any, policy_obs: dict[str, torch.Tensor]) -> None:
        quat = torch.tensor(self.spec.carry_quat_world, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        if self.attached:
            offset = torch.tensor(self.spec.hold_offset, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
            object_pos_b = policy_obs["eef_pos"].reshape(env.num_envs, 3) + offset
            set_rigid_pose(env, self.spec.object_name, base_to_world(env, object_pos_b), quat)
        elif self.released:
            target_w = torch.tensor(self.spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(
                env.num_envs, 1
            )
            set_rigid_pose(env, self.spec.object_name, target_w, quat)


def rollout_success(env: Any, spec: TaskSpec) -> bool:
    object_pos = env.scene[spec.object_name].data.root_pos_w
    target = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    dist = torch.linalg.vector_norm(object_pos[:, :2] - target[:, :2], dim=-1)
    z_ok = torch.abs(object_pos[:, 2] - target[:, 2]) < 0.08
    return bool(((dist < 0.08) & z_ok).all().item())


def mobile_rollout_success(env: Any, spec: MobileTaskSpec) -> bool:
    object_pos = env.scene[spec.base.object_name].data.root_pos_w
    target = torch.tensor(spec.target_world_pos, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
    dist = torch.linalg.vector_norm(object_pos[:, :2] - target[:, :2], dim=-1)
    z_ok = torch.abs(object_pos[:, 2] - target[:, 2]) < 0.08
    return bool(((dist < 0.08) & z_ok).all().item())
