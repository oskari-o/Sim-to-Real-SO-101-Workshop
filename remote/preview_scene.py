"""Preview an SO-101 workshop task in Isaac Lab without a leader arm.

Loads the task's scene (robot, vials, rack, cameras, lighting), holds the robot
still at its start pose, and resets every --reset_every seconds so you can see
the domain randomization reshuffle the scene.

Run inside the teleop container, from the repo root:
    /workspace/isaaclab/_isaac_sim/python.sh outputs/preview_scene.py \
        --task Lerobot-So101-Teleop-Vials-To-Rack-DR --livestream 2 \
        --kit_args "--/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP>"
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview a workshop task without teleop hardware.")
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--reset_every", type=float, default=15.0, help="seconds between resets, 0 = never")
parser.add_argument("--disable_fabric", action="store_true", help="use USD I/O instead of Fabric")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

# Everything below needs the simulator running.
import importlib  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Register the workshop's tasks. The exact module isn't documented, so try the likely ones.
for mod in ("sim_to_real_so101.tasks", "sim_to_real_so101"):
    try:
        importlib.import_module(mod)
    except ImportError as e:
        print(f"[preview] could not import {mod}: {e}")
if args.task not in gym.registry:
    so101 = [k for k in gym.registry if "So101" in k or "so101" in k.lower()]
    raise SystemExit(f"[preview] task {args.task!r} not registered. Registered SO-101 tasks: {so101}")

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
env = gym.make(args.task, cfg=env_cfg)
env.reset()
base = env.unwrapped
print(f"[preview] loaded {args.task}; action space {env.action_space}")


def hold_action():
    """Action that keeps the arm where it is (assumes absolute joint-position actions)."""
    try:
        robot = base.scene["robot"]
        act = robot.data.joint_pos[:, : env.action_space.shape[-1]].clone()
        return act.to(base.device)
    except Exception:
        return torch.zeros(env.action_space.shape, device=base.device)


step_dt = base.step_dt
steps_per_reset = int(args.reset_every / step_dt) if args.reset_every > 0 else 0
action = hold_action()
step = 0
while app.is_running():
    with torch.inference_mode():
        env.step(action)
        step += 1
        if steps_per_reset and step % steps_per_reset == 0:
            env.reset()
            action = hold_action()
            print("[preview] reset (new randomization)")

env.close()
app.close()
