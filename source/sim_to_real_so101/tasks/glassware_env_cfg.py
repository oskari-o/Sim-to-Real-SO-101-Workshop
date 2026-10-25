# Glassware scene: the course's robot, lightbox, cameras and mat, with only two props on the
# mat, the transparent PET bottle and the cut-crystal tumbler. No vials, no rack, no success
# terms; teleop and recording work the same as in the other tasks.
#
# Assets (generate them first, see BOTTLE_SCENE.md):
#   asset_gen/make_bottle.py  -> source/sim_to_real_so101/assets/usd/Bottle.usda
#   asset_gen/make_tumbler.py -> source/sim_to_real_so101/assets/usd/Tumbler.usda
#
# Optional fill, same environment variables as the Vials-Bottle tasks (BOTTLE_FILL=none|beads|pbd,
# BOTTLE_FILL_FRACTION, ...), plus:
#   FILL_TARGET=bottle|tumbler   which glass gets the beads / liquid (default: bottle)
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from sim_to_real_so101 import assets
from sim_to_real_so101.mdp import (
    ROBOT_COLORS,
    randomize_mat_rotation,
    randomize_robot_color,
    randomize_sky_light,
)

from .task_env_cfg import SO101TaskEnvCfg, SO101TaskSceneCfg, TaskEventCfg
from .vials_bottle_env_cfg import BOTTLE_CFG, BOTTLE_USD, apply_bottle_fill

assets_path = os.path.dirname(os.path.abspath(assets.__file__))

# Positions on the mat (mat surface at z ~0.032; vials used to stand around x = 0.23).
# Both assets have their origin at the centre of their underside, so z is a small drop height.
GLASS_BOTTLE_POS = (0.22, -0.10, 0.045)
TUMBLER_POS = (0.20, 0.09, 0.037)
TUMBLER_USD = f"{assets_path}/usd/Tumbler.usda"

TUMBLER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Tumbler",
    # mass, COM, inertia, collider and friction come from the USD (make_tumbler.py)
    spawn=sim_utils.UsdFileCfg(
        usd_path=TUMBLER_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(angular_damping=2.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=TUMBLER_POS),
)


def _reset_to_start(name: str) -> EventTerm:
    """Put a prop back to its start pose on every reset (no randomization)."""
    return EventTerm(
        func=isaaclab_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg(name)},
    )


@configclass
class GlasswareSceneCfg(SO101TaskSceneCfg):
    bottle = BOTTLE_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=GLASS_BOTTLE_POS))
    tumbler = TUMBLER_CFG.replace()


@configclass
class GlasswareDRSceneCfg(GlasswareSceneCfg):
    sky_light = AssetBaseCfg(
        prim_path="/World/sky_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{assets_path}/hdri/moon_lab_1k.exr",
            visible_in_primary_ray=False,
            enable_color_temperature=True,
            color_temperature=6500.0,
        ),
    )


@configclass
class GlasswareEventCfg(TaskEventCfg):
    reset_bottle = _reset_to_start("bottle")
    reset_tumbler = _reset_to_start("tumbler")


@configclass
class GlasswareEventDRCfg(GlasswareEventCfg):
    """Same domain randomization as the vials DR task: robot colour, HDRI sky, wider mat rotation."""

    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": list(ROBOT_COLORS.keys())},
    )
    reset_sky_light = EventTerm(
        func=randomize_sky_light,
        mode="reset",
        params={
            "exposure_range": (-4.0, 3.0),
            "temperature_range": (2500.0, 9500.0),
            "textures_root": f"{assets_path}/hdri",
            "asset_cfg": SceneEntityCfg("sky_light"),
        },
    )
    reset_mat_rotation = EventTerm(
        func=randomize_mat_rotation,
        mode="reset",
        params={"yaw_range": (-0.3, 0.3), "asset_cfg": SceneEntityCfg("mat")},
    )


def _apply_fill(env_cfg) -> None:
    target = os.getenv("FILL_TARGET", "bottle").strip().lower()
    if target == "tumbler":
        apply_bottle_fill(env_cfg, TUMBLER_USD, TUMBLER_POS)
    elif target == "bottle":
        apply_bottle_fill(env_cfg, BOTTLE_USD, GLASS_BOTTLE_POS)
    else:
        raise ValueError(f"FILL_TARGET must be bottle or tumbler, got {target!r}")


@configclass
class GlasswareEnvCfg(SO101TaskEnvCfg):
    scene: GlasswareSceneCfg = GlasswareSceneCfg()
    events: GlasswareEventCfg = GlasswareEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()  # also enables translucency + refraction settings for the glass
        _apply_fill(self)


@configclass
class GlasswareDREnvCfg(SO101TaskEnvCfg):
    scene: GlasswareDRSceneCfg = GlasswareDRSceneCfg()
    events: GlasswareEventDRCfg = GlasswareEventDRCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_fill(self)
