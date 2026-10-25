"""Set up the Kit viewports (cameras, resolution/aspect, a docked second viewport) at startup.

Kit's saved layouts (Window > Layout > Save Layout) only keep where windows are docked, not which
camera a viewport looks through or its resolution, and the stage is rebuilt on every run. So the
scripts call setup_viewports() after the env is created, configured with one environment variable:

  VIEWPORTS="persp,ego@640x480"
  VIEWPORTS="external_D455@640x480,ego@640x480"

Comma-separated, one entry per viewport (1st = the main "Viewport", 2nd and later are created and
docked to its right). Each entry is a camera, optionally followed by @WIDTHxHEIGHT:
  - a scene camera name: "ego", "external_D455" (or "camera_ego", ...), env 0's camera is used
  - "persp" / "top" / "front" / "right": Kit's built-in cameras
  - an absolute prim path, e.g. /World/envs/env_0/Robot/gripper/gripper_cam
The resolution sets the viewport's render resolution, and with it the aspect ratio.
"""
from __future__ import annotations

import os
import re

BUILTIN = {"persp": "/OmniverseKit_Persp", "top": "/OmniverseKit_Top", "front": "/OmniverseKit_Front",
           "right": "/OmniverseKit_Right"}


def _camera_path(scene, name: str) -> str:
    if name.startswith("/"):
        return name
    if name in BUILTIN:
        return BUILTIN[name]
    key = name if name.startswith("camera_") else f"camera_{name}"
    cfg = getattr(scene.cfg, key, None)
    if cfg is None:
        cams = [k for k in scene.keys() if k.startswith("camera_")]
        raise ValueError(f"unknown camera {name!r}; scene cameras: {cams}, or persp/top/front/right, or a prim path")
    return re.sub(r"env_\.\*", "env_0", cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"))


def setup_viewports(env, spec: str | None = None, pump_frames: int = 3) -> None:
    """Apply a VIEWPORTS spec (see module doc). Never raises: problems are printed and skipped."""
    spec = spec if spec is not None else os.getenv("VIEWPORTS", "")
    if not spec.strip():
        return
    try:
        import omni.kit.app
        import omni.ui as ui
        from omni.kit.viewport.utility import create_viewport_window, get_active_viewport_window
    except ImportError as e:
        print(f"[viewports] viewport utilities not available ({e}); skipping")
        return
    scene = env.unwrapped.scene
    app = omni.kit.app.get_app()
    main = get_active_viewport_window()
    windows = [main]
    for i, entry in enumerate(e.strip() for e in spec.split(",") if e.strip()):
        name, _, res = entry.partition("@")
        try:
            cam = _camera_path(scene, name.strip())
            if i == 0:
                win = main
            else:
                win = create_viewport_window(f"Viewport {i + 1}", width=640, height=480)
                for _ in range(pump_frames):
                    app.update()
                target = ui.Workspace.get_window(windows[-1].title) if windows[-1] else None
                if target is not None:
                    win.dock_in(target, ui.DockPosition.RIGHT, 0.5)
                windows.append(win)
            win.viewport_api.camera_path = cam
            if res:
                w, h = (int(v) for v in res.lower().split("x"))
                win.viewport_api.resolution = (w, h)
            print(f"[viewports] {win.title}: {cam}" + (f" @ {res}" if res else ""))
        except Exception as e:  # noqa: BLE001  never break a run over viewport cosmetics
            print(f"[viewports] could not set up {entry!r}: {e}")
    for _ in range(pump_frames):
        app.update()
