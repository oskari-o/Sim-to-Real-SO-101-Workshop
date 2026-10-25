#!/bin/bash
# Livestream settings for Isaac Sim / Isaac Lab runs in the teleop container.
# Source it (don't run it), then pass "$KIT" to --kit_args:
#   source remote/kit_env.sh
#   /workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py --task ... --livestream 2 --kit_args "$KIT"
#
# SERVER_IP: the server's public IPv4. Set it yourself, or it's detected (falls back to the fixed one).
SERVER_IP="${SERVER_IP:-$(curl -4 -s --max-time 3 ifconfig.me 2>/dev/null)}"
SERVER_IP="${SERVER_IP:-94.237.11.153}"
export SERVER_IP
export KIT="--/exts/omni.kit.livestream.app/primaryStream/publicIp=${SERVER_IP} --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998"
export PYTHONUNBUFFERED=1
# Viewport layout applied at startup (utils/viewports.py): main viewport + a docked second one.
# Entries: scene camera (ego, external_D455), persp/top/front/right, or a prim path; optional @WxH.
export VIEWPORTS="${VIEWPORTS:-external_D455@640x480,ego@640x480}"
echo "KIT set for ${SERVER_IP} (signal 49100/tcp, stream 47998/udp); VIEWPORTS=${VIEWPORTS}"
