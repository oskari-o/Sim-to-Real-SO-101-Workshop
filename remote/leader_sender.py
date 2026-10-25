#!/usr/bin/env python3
"""Read the SO-101 leader arm locally and stream its joint positions to the remote sim.

Sturdier than forwarding the serial port (serial_bridge.py + socat): the motor bus is
polled here on the Mac, at full USB speed, and only finished joint positions cross the
network, one way. A late packet just means a slightly older pose, never a crash.

Run on the Mac, in a venv with LeRobot installed and the leader calibrated:
    python remote/leader_sender.py --port /dev/cu.usbmodemXXXX --id <calibration id>
    ssh -N -R 54322:localhost:54322 upcloud_gpu
and on the server run lerobot_agent.py with --teleop_device virtual (same as the slider app;
the two can't run at the same time since they use the same port).

Type r / s / c + Enter here for reset world / start-stop recording / cancel episode.

--id is the name of the leader's calibration file, without .json. Newer LeRobot keeps
it in ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/, older in so101_leader/.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def make_leader(port: str, leader_id: str):
    """Create the leader with whichever LeRobot API is installed, in -100..100 units."""
    try:  # LeRobot 2026+: generic SO leader
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    except ImportError:  # older LeRobot (what the sim container uses)
        from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    kwargs = {"port": port, "id": leader_id}
    fields = getattr(SO101LeaderConfig, "__dataclass_fields__", {})
    if "use_degrees" in fields:
        kwargs["use_degrees"] = False  # sim mapping expects -100..100, not degrees
    if "num_read_retries" in fields:
        kwargs["num_read_retries"] = 3
    return SO101Leader(SO101LeaderConfig(**kwargs))


class Link:
    """TCP server the sim's VirtualLeader connects to (through the ssh -R tunnel)."""

    def __init__(self, port: int, max_step: float):
        self.max_step = max_step
        self.conn: socket.socket | None = None
        self.lock = threading.Lock()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        self.srv = srv
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            c, _ = self.srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                if self.conn:
                    self.conn.close()
                self.conn = c
            print("\n[sender] simulator connected")
            self.send({"type": "config", "max_step": self.max_step})
            threading.Thread(target=self._drain, args=(c,), daemon=True).start()

    def _drain(self, c):
        """Read (and mostly ignore) what the sim sends; notice disconnects and recording state."""
        buf = b""
        try:
            while True:
                chunk = c.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("type") == "status":
                        print(f"\n[sender] recording {'STARTED' if msg.get('recording') else 'stopped'}")
        except OSError:
            pass
        with self.lock:
            if self.conn is c:
                self.conn = None
        print("\n[sender] simulator disconnected; waiting for it to reconnect")

    def send(self, msg: dict) -> bool:
        with self.lock:
            c = self.conn
        if c is None:
            return False
        try:
            c.sendall((json.dumps(msg) + "\n").encode())
            return True
        except OSError:
            return False


def stdin_commands(link: Link):
    keys = {"r": "reset", "s": "record", "c": "cancel"}
    for line in sys.stdin:
        cmd = keys.get(line.strip().lower())
        if cmd:
            print(f"[sender] -> {cmd}" if link.send({"type": "cmd", "cmd": cmd}) else "[sender] sim not connected")
        elif line.strip():
            print("[sender] commands: r = reset world, s = start/stop recording, c = cancel episode")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="leader serial port, e.g. /dev/cu.usbmodem5A7A0185421")
    ap.add_argument("--id", required=True, help="leader calibration id (calibration file name without .json)")
    ap.add_argument("--sim-port", type=int, default=54322, help="TCP port the sim connects to")
    ap.add_argument("--hz", type=float, default=60.0, help="read/send rate")
    ap.add_argument("--max-step", type=float, default=10.0,
                    help="sim-side rate limit per step in leader units (10 ~ 600 units/s; 0 = off). "
                         "Only matters when the sim pose is far from the leader, e.g. right after connecting.")
    a = ap.parse_args()

    leader = make_leader(a.port, a.id)
    leader.connect()
    print(f"[sender] leader connected on {a.port} (id {a.id})")
    link = Link(a.sim_port, a.max_step)
    print(f"[sender] waiting for the sim on 127.0.0.1:{a.sim_port} "
          f"(tunnel: ssh -N -R {a.sim_port}:localhost:{a.sim_port} upcloud_gpu)")
    print("[sender] type r / s / c + Enter: reset world / start-stop recording / cancel episode")
    threading.Thread(target=stdin_commands, args=(link,), daemon=True).start()

    period, fails, last_print = 1.0 / a.hz, 0, 0.0
    try:
        while True:
            t0 = time.perf_counter()
            try:
                act = leader.get_action()
                fails = 0
                link.send({"type": "target", "action": {j: float(act[j]) for j in JOINTS}})
                if t0 - last_print > 0.5:
                    vals = " ".join(f"{act[j]:6.1f}" for j in JOINTS)
                    print(f"\r[sender] {vals}   ", end="", flush=True)
                    last_print = t0
            except Exception as e:  # noqa: BLE001  keep going on transient bus errors
                fails += 1
                if fails in (1, 10) or fails % 100 == 0:
                    print(f"\n[sender] read error ({fails}x): {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            leader.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
