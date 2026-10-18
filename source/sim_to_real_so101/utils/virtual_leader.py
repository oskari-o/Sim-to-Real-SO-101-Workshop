"""A software stand-in for the SO-101 leader arm.

`VirtualLeader.get_action()` returns the same dict as LeRobot's SO101Leader
(`shoulder_pan.pos` ... `gripper.pos`, arm joints in -100..100, gripper in 0..100),
so lerobot_agent.py's mapping and recording code work unchanged.

The targets come from remote/joint_teleop_app.py running on your local machine.
That app listens on TCP (default 127.0.0.1:54322); expose it to the server with
`ssh -N -R 54322:localhost:54322 upcloud_gpu`, and this class connects to it.

Wire protocol: newline-delimited JSON, both ways.
  app -> sim : {"type": "target", "action": {"shoulder_pan.pos": 0.0, ...}}
               {"type": "cmd", "cmd": "reset" | "record" | "cancel"}
  sim -> app : {"type": "state", "action": {...}, "home": {...}}   (on connect and after resets)
               {"type": "status", "recording": bool}
"""
from __future__ import annotations

import json
import socket
import threading
import time

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
LIMITS = {j: (-100.0, 100.0) for j in JOINTS}
LIMITS["gripper.pos"] = (0.0, 100.0)


class VirtualLeader:
    """Drop-in for `robot_iface.robot` when no physical leader arm is attached."""

    def __init__(self, home: dict, host: str = "127.0.0.1", port: int = 54322,
                 max_step: float = 2.0):
        """
        Args:
            home: start pose in leader units (what the sim robot is at after reset).
            max_step: max change per get_action() call, in leader units. The agent
                calls it once per env step (60 Hz here), so 2.0 ~= 120 units/s, which
                is roughly 130 deg/s on most joints. Keeps slider jumps from
                slamming the arm. Set <= 0 to disable.
        """
        self.name = "virtual_leader"
        self.host, self.port, self.max_step = host, port, max_step
        self._home = {j: float(home[j]) for j in JOINTS}
        self._target = dict(self._home)
        self._current = dict(self._home)
        self._lock = threading.Lock()
        self._cmds: list[str] = []
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self.is_connected = False
        threading.Thread(target=self._run, daemon=True).start()

    # ----------------------------------------------------------- LeRobot-like API

    def connect(self, *_, **__):
        pass

    def disconnect(self):
        self._stop.set()
        if self._sock:
            self._sock.close()

    def get_action(self) -> dict:
        with self._lock:
            for j in JOINTS:
                d = self._target[j] - self._current[j]
                if self.max_step > 0:
                    d = max(-self.max_step, min(self.max_step, d))
                self._current[j] += d
            return dict(self._current)

    # ----------------------------------------------------------- agent hooks

    def pop_commands(self) -> list[str]:
        with self._lock:
            cmds, self._cmds = self._cmds, []
        return cmds

    def sync(self, pose: dict | None = None):
        """Snap target and current to `pose` (default: home), and tell the app."""
        pose = {j: float((pose or self._home)[j]) for j in JOINTS}
        with self._lock:
            self._target, self._current = dict(pose), dict(pose)
        self._send({"type": "state", "action": pose, "home": self._home})

    def send_status(self, recording: bool):
        self._send({"type": "status", "recording": bool(recording)})

    # ----------------------------------------------------------- networking

    def _send(self, msg: dict):
        s = self._sock
        if s is None:
            return
        try:
            s.sendall((json.dumps(msg) + "\n").encode())
        except OSError:
            pass

    def _run(self):
        warned = False
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
            except OSError:
                if not warned:
                    print(f"[VirtualLeader] waiting for teleop app on {self.host}:{self.port} "
                          "(is the app running and the ssh -R tunnel up?)")
                    warned = True
                time.sleep(1.0)
                continue
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)
            self._sock, self.is_connected, warned = s, True, False
            print(f"[VirtualLeader] connected to teleop app on {self.host}:{self.port}")
            with self._lock:
                cur = dict(self._current)
            self._send({"type": "state", "action": cur, "home": self._home})
            buf = b""
            try:
                while not self._stop.is_set():
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            self._handle(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                print(f"[VirtualLeader] connection error: {e}")
            finally:
                self._sock, self.is_connected = None, False
                s.close()
                print("[VirtualLeader] disconnected; holding last pose")

    def _handle(self, msg: dict):
        kind = msg.get("type")
        if kind == "target":
            act = msg.get("action", {})
            with self._lock:
                for j in JOINTS:
                    if j in act:
                        lo, hi = LIMITS[j]
                        self._target[j] = max(lo, min(hi, float(act[j])))
        elif kind == "cmd":
            cmd = msg.get("cmd")
            if cmd in ("reset", "record", "cancel"):
                with self._lock:
                    self._cmds.append(cmd)
