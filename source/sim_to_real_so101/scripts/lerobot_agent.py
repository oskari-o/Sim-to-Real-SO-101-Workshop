# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Teleop agent.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Port of the robot.",
)
parser.add_argument(
    "--robot_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="ID of the robot.",
)
parser.add_argument(
    "--repo_id", type=str, default=None, help="Repository ID to store the dataset."
)
parser.add_argument(
    "--repo_root", type=str, default=None, help="Repository root to store the dataset."
)
parser.add_argument(
    "--save_mp4",
    action="store_true",
    default=False,
    help="Save depth and RGB as mp4 videos.",
)
parser.add_argument(
    "--depth", action="store_true", default=False, help="Save depth as mp4 video."
)
parser.add_argument(
    "--instance_id_seg",
    action="store_true",
    default=False,
    help="Save instance id segmentation as mp4 video.",
)
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=101, help="Environment seed")
parser.add_argument(
    "--teleop_device",
    choices=["leader", "virtual"],
    default=os.getenv("TELEOP_DEVICE", "leader"),
    help="'leader' = physical SO-101 leader on --port; "
    "'virtual' = joint sliders from remote/joint_teleop_app.py over TCP.",
)
parser.add_argument(
    "--virtual_host", type=str, default="127.0.0.1", help="Host of the joint teleop app."
)
parser.add_argument(
    "--virtual_port",
    type=int,
    default=int(os.getenv("VIRTUAL_TELEOP_PORT", "54322")),
    help="TCP port of the joint teleop app (forwarded with ssh -R).",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import time


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder
from sim_to_real_so101.utils.virtual_leader import VirtualLeader


def main():

    keyboard_control = KeyboardControl()

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")
    print(f"[INFO]: Click 'S' to start/stop recording; 'R' will also stop recording")

    # reset environment
    obs, _ = env.reset()

    # cameras
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print(f"[Info]: No cameras found - videos will not be recorded")

    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras=cameras,
        fps=30,
        kind="leader",
    )
    if args_cli.teleop_device == "virtual":
        # No hardware: targets come from the joint teleop app. Start from the sim's reset pose.
        home = robot_iface.get_raw_actions_from_radians(obs["policy"]["joint_pos_obs"][0])
        home = dict(zip(robot_iface.SO101_JOINT_ORDER, home.tolist()))
        robot_iface.robot = VirtualLeader(
            home, host=args_cli.virtual_host, port=args_cli.virtual_port
        )
        print(f"[INFO]: Using virtual leader (joint teleop app on {args_cli.virtual_host}:{args_cli.virtual_port})")
    else:
        robot_iface.init_device()
        robot_iface.connect()
    virtual = isinstance(robot_iface.robot, VirtualLeader)
    last_real_action, read_failures = None, 0
    was_recording = False

    # Allocate action tensor
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # simulate environment

    # Recording dataset
    if all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name]):
        recording_mode = True
    else:
        recording_mode = False

    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=args_cli.save_mp4,
            depth=args_cli.depth,
            instance_id_seg=args_cli.instance_id_seg,
        )
        try:
            recorder.init_dataset()
        except ValueError:
            print(f"[ERROR]: Failed to initialize dataset. folder already exists")
            env.close()
            simulation_app.close()

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            try:
                real_action = robot_iface.robot.get_action()
                last_real_action, read_failures = real_action, 0
            except ConnectionError as e:
                # A leader read over a slow link (e.g. serial forwarded over SSH) can time out.
                # Hold the last pose instead of crashing; give up only if it keeps failing.
                read_failures += 1
                if last_real_action is None or read_failures > 120:
                    raise
                if read_failures == 1 or read_failures % 30 == 0:
                    print(f"[WARN]: leader read failed ({read_failures}x), holding last pose: {e}")
                try:
                    robot_iface.robot.bus.port_handler.clearPort()  # drop late replies
                except Exception:  # noqa: BLE001
                    pass
                real_action = last_real_action
            real_action, mapped_action = robot_iface.real_to_sim_obs_processor(
                real_action
            )
            actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)

            if virtual:
                # Buttons in the teleop app mirror the R / S / C keys
                for cmd in robot_iface.robot.pop_commands():
                    if cmd == "reset":
                        keyboard_control.reset_world = True
                        keyboard_control.stop_recording()
                    elif cmd == "record":
                        if keyboard_control.recording:
                            keyboard_control.stop_recording()
                        else:
                            keyboard_control.start_recording()
                    elif cmd == "cancel":
                        keyboard_control.cancel_recording()
                if keyboard_control.recording != was_recording:
                    was_recording = keyboard_control.recording
                    robot_iface.robot.send_status(was_recording)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                if virtual:
                    robot_iface.robot.sync()  # sliders jump back to the start pose
                continue

            if recording_mode and keyboard_control.recording:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print(
                        "[WARNING]: No 'visual' observation group - recording requires a task with cameras"
                    )
                    keyboard_control.recording = False
                    continue
                # Extract joint positions from policy observation dict
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                visual_obs = obs["visual"]
                real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                    robot_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
                )
                recorder.push_frame_to_buffer(
                    real_action,
                    real_obs,
                    visual_buffers,
                    depth_buffers,
                    instance_id_seg_buffers,
                )

    env.close()


if __name__ == "__main__":

    main()

    while True:
        simulation_app.update()

    simulation_app.close()
