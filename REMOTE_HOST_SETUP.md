# Running the SO-101 Sim-to-Real course on a remote GPU host

This fork runs the course's simulation and GPU work on a rented cloud GPU (tested on UpCloud, Helsinki, 1x L40S) while you work from a Mac. Isaac Sim runs headless in Docker on the server and is streamed to the Mac with WebRTC. The SO-101 arms stay plugged into the Mac, and the leader arm can be forwarded to the server over SSH. Without the leader arm, a browser-based joint-slider app on the Mac can drive the sim instead (step 10b).

The fork also adds a transparent bottle prop and `Vials-Bottle` tasks; see [BOTTLE_SCENE.md](BOTTLE_SCENE.md).

Helper files in [`remote/`](remote/):

| File | Runs on | Purpose |
| --- | --- | --- |
| `run_teleop.sh` | server | Starts `teleop-docker` without X11, with all driver capabilities |
| `preview_scene.py` | server, in container | Loads a course task with no arm connected, resets periodically |
| `serial_bridge.py` | Mac | Exposes the leader arm's serial port over TCP |
| `joint_teleop_app.py` | Mac | Joint-slider teleop UI in the browser, used with `lerobot_agent.py --teleop_device virtual` |

Placeholders used below: `<SERVER_IP>` is the server's public IPv4, `<HOME_IP>` is your own public IPv4 (`curl -4 ifconfig.me` on the Mac).

---

## 1. Rent the server

UpCloud Hub → Deploy GPU Server:

- **Location:** FI-HEL2 (low latency from Finland matters for streaming and teleop).
- **Pricing:** On-demand. Spot can be reclaimed mid-session.
- **Plan:** 1x L40S, 12 cores / 128 GB (about €1.25/h). Isaac Sim plus GR00T plus builds squeeze 64 GB.
- **Storage:** 150–250 GB MaxIOPS. 50 GB default is far too small. Volumes are easy to grow and hard to shrink.
- **Backups:** off. Everything is reproducible.
- **OS:** Ubuntu Server 24.04 LTS. The "with NVIDIA drivers & CUDA" template ships a 595-branch, compute-only driver, which does **not** work for Isaac Sim 5.1 (see step 5). Either template works with the steps below; the plain one saves the purge.
- **Network:** keep public IPv4 on.
- **Login:** SSH key (step 2).
- **GPU choice in general:** needs RT cores and NVENC, so L4 / L40S / RTX 4090 / RTX 6000 Ada / RTX PRO 6000. Not A100, H100, B200 (no RT cores). Use build target `ada` for Ada cards, `blackwell` for RTX PRO 6000.

Billing: a running server bills even when idle. A stopped server bills only for storage and the public IP. Shut it down from the Hub after each session. Enable 2FA on the account.

## 2. SSH key and config (Mac)

```bash
ssh-keygen -t ed25519 -C "upcloud" -f ~/.ssh/upcloud_so101
pbcopy < ~/.ssh/upcloud_so101.pub   # paste into "Add new" SSH key on the deploy page
```

After deploy, add to `~/.ssh/config`:

```
Host upcloud_gpu
  HostName <SERVER_IP>
  User root
  IdentityFile ~/.ssh/upcloud_so101
  UseKeychain yes
  AddKeysToAgent yes
  ForwardAgent yes
```

`ForwardAgent yes` lets the server use the Mac's GitHub SSH key, so you can clone and push the fork over SSH without keeping a key on the server. To edit on the server from the Mac, VS Code's Remote-SSH extension can connect to the `upcloud_gpu` host directly.

UpCloud's templates put the key on `root`, not `ubuntu`. The first connection asks to confirm the host key; type `yes` in full.

## 3. Firewall (Hub → server → Firewall)

Incoming rules, in order:

1. Accept TCP from `<HOME_IP>`, all ports
2. Accept UDP from `<HOME_IP>`, all ports
3. Optional: accept ICMP from `<HOME_IP>`

Tick **Auto-add these DNS rules** (otherwise apt, Docker pulls and downloads hang). Default incoming rule: drop. Outgoing: accept.

The livestream has no authentication or encryption, so never open 49100/TCP and 47998/UDP to the world. Use UpCloud's firewall rather than `ufw`: Docker bypasses `ufw` for published ports.

When SSH suddenly times out, your home IP changed. Update both rules.

## 4. Base packages, Docker, NVIDIA Container Toolkit

```bash
ssh upcloud_gpu
apt update && apt upgrade -y
apt install -y tmux git git-lfs pipx

curl -fsSL https://get.docker.com | sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update && apt install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

## 5. NVIDIA driver: use the 580 branch

Isaac Sim 5.1 is validated on the 580 driver branch. 595.x drivers crash at startup with a segfault in `librtx.scenedb.plugin.so` (a known issue confirmed on NVIDIA's forum). Isaac Sim also needs the full driver, including the graphics, Vulkan, ray-tracing and NVENC libraries, not a compute-only install.

Ubuntu's `nvidia-driver-580-server-open` is a complete package set. Install it from Ubuntu's archive and keep NVIDIA's CUDA apt repo out of the picture, because mixing the two removes pieces of the driver.

**If you used UpCloud's NVIDIA template, purge its driver first:**

```bash
mkdir -p /root/apt-disabled
mv /etc/apt/sources.list.d/cuda-*.list /root/apt-disabled/ 2>/dev/null
mv /etc/apt/preferences.d/cuda-repository-pin-600 /root/apt-disabled/ 2>/dev/null

PKGS=$(dpkg -l | awk '/^ii/ && $2 ~ /nvidia|cuda-drivers/ && $2 !~ /container/ {print $2}')
echo $PKGS          # check: driver/library packages only, nothing *container*
apt purge -y $PKGS
apt autoremove -y
mv /etc/apt/preferences.d/nvidia-driver-pin /root/apt-disabled/ 2>/dev/null
```

**Install 580 (both templates):**

```bash
apt update
apt-cache policy nvidia-driver-580-server-open | head -5   # candidate: 580.x from an Ubuntu mirror
apt install -y linux-headers-$(uname -r) nvidia-driver-580-server-open

dkms status                                                # nvidia 580.x installed
ldconfig -p | grep -E 'libnvidia-encode|libGLX_nvidia|libnvidia-rtcore'   # all three present
reboot
```

After reconnecting:

```bash
nvidia-smi                                   # L40S, driver 580.x
docker run --rm --gpus all ubuntu nvidia-smi # GPU visible from Docker
```

Do not `apt install nvidia-utils-*` or `libnvidia-*-595` from suggestions; they pull mismatched versions.

## 6. Code, FFmpeg fix, containers

```bash
git config --global user.name "<your name>"
git config --global user.email "<your email>"
git clone git@github.com:<you>/Sim-to-Real-SO-101-Workshop.git ~/Sim-to-Real-SO-101-Workshop
cd ~/Sim-to-Real-SO-101-Workshop
git remote add upstream <NVIDIA's original repo URL>   # to pull course updates later
git lfs install && git lfs pull
chmod +x remote/run_teleop.sh
```

**FFmpeg URL fix.** `docker/sim/Dockerfile` downloads FFmpeg 7.1 from BtbN's rolling `latest` release, which no longer has 7.1 (404). The pinned `torchcodec<0.6` needs FFmpeg 7, so 8.x is not an option. Point it at a dated build instead (commit this in the fork):

```bash
curl -s "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases?per_page=100" \
  | grep -o 'https://[^"]*ffmpeg-n7\.1[^"]*linux64-lgpl-shared[^"]*\.tar\.xz' | head -3
# pick the newest URL, then:
sed -i 's#https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz#<DATED_URL>#' docker/sim/Dockerfile
```

Used successfully: `https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-lgpl-shared-7.1.tar.xz`. Dated releases also get pruned eventually; to be fully safe, keep a copy of the tarball and point the Dockerfile at it.

**Build, in tmux** (`tmux new -s build`; detach with Ctrl-b d, reattach with `tmux attach -t build`):

```bash
docker build -t teleop-docker -f docker/sim/Dockerfile .
./docker/real/build.sh ada        # about an hour; only needed from the GR00T modules on
```

**Models** (as in the course's "Get the Code" page):

```bash
pipx ensurepath && pipx install huggingface_hub   # then open a new shell
hf auth login
# then the course's mkdir -p models and hf download commands, from the repo root
```

## 7. Mac: streaming client

Install the **Isaac Sim WebRTC Streaming Client** for macOS from the Isaac Sim 5.1 download page (Latest Release section). Open the DMG and drag it to Applications.

## 8. Smoke test: plain Isaac Sim over WebRTC

```bash
tmux new -s sim
~/Sim-to-Real-SO-101-Workshop/remote/run_teleop.sh
# now inside the container (prompt shows /workspace/Sim-to-Real-SO-101-Workshop)
ldconfig -p | grep libnvidia-encode      # must print a line
cd /isaac-sim
./runheadless.sh \
  --/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP> \
  --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 \
  --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998
```

Wait for `Isaac Sim Full Streaming App is loaded` (first start compiles shaders, a few minutes), then connect the Mac client to `<SERVER_IP>`. Quit with File > Exit in the streamed app.

The `groups: cannot find name for group ID ...` line at container start is harmless.

## 9. Preview the course scene without the arm

`lerobot_agent` connects to the leader arm by default and fails without it (`ConnectionError` on the serial port). To drive the sim without the arm, use the joint sliders in step 10b. To just look at a scene, `remote/preview_scene.py` loads the same task, holds the robot still and resets every 15 s to show the domain randomization. Inside the container, from `/workspace/Sim-to-Real-SO-101-Workshop`:

```bash
/workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py \
  --task Lerobot-So101-Teleop-Vials-To-Rack-DR --livestream 2 --enable_cameras \
  --kit_args "--/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP> --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998" \
  2>&1 | tee outputs/preview.log
```

`--enable_cameras` is required because the task has cameras. Options: `--task Lerobot-So101-Teleop-Vials-To-Rack` (no DR), `--task Lerobot-So101-Teleop-Vials-Bottle-DR` (with the bottle, see [BOTTLE_SCENE.md](BOTTLE_SCENE.md)), `--reset_every 0`.

Viewport navigation in the client (click the viewport first; Option = Alt): Alt + left-drag orbit, middle-drag pan, scroll zoom, right-click + WASD/QE fly, select a prim in the Stage panel and press F to frame it. Navigate in the Perspective camera, not in `gripper_cam` or `Camera_OmniVision_9782_Color`, or you move those cameras.

## 10. Teleop into the remote sim with the leader arm on the Mac

The arm's serial port is forwarded Mac → SSH tunnel → `socat` virtual port in the container. Start in this order each session.

**Mac, terminal 1:** bridge

```bash
ls /dev/cu.usbmodem*
pip3 install pyserial
python3 remote/serial_bridge.py /dev/cu.usbmodemXXXX
```

**Mac, terminal 2:** reverse tunnel

```bash
ssh -N -R 54321:localhost:54321 upcloud_gpu
```

**Mac, once:** copy the leader arm calibration to the server (filename = your teleop id)

```bash
ssh upcloud_gpu 'mkdir -p ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader'
scp ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/orange_teleop.json \
  upcloud_gpu:~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/
```

**Server, second tmux window:** virtual serial port inside the container

```bash
docker exec -it teleop bash
apt-get update && apt-get install -y socat
socat -d -d pty,raw,echo=0,link=/dev/ttyACM1 tcp:localhost:54321,nodelay
```

**Server, container shell:** run the course command with streaming added

```bash
export TELEOP_PORT=/dev/ttyACM1 TELEOP_ID=orange_teleop
lerobot_agent --task Lerobot-So101-Teleop-Vials-To-Rack-DR --livestream 2 \
  --kit_args "--/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP> --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998" \
  2>&1 | tee outputs/agent_run.log
```

Recording works the same way with the course's `--repo_id`, `--repo_root` and `--task_name` arguments added. `socat` is gone after a container restart; redo that step.

## 10b. Teleop without a leader arm: joint sliders

`remote/joint_teleop_app.py` serves a page with one slider per joint, hold-to-move keys and record/reset buttons. On the server, `lerobot_agent.py --teleop_device virtual` replaces the leader arm with `utils/virtual_leader.py`. It returns the same values a leader arm reports, so the mapping and LeRobot recording are unchanged.

**Mac, terminal 1:** the app (standard library only, no venv needed)

```bash
python3 remote/joint_teleop_app.py      # then open http://127.0.0.1:8765
```

**Mac, terminal 2:** reverse tunnel for the app's port

```bash
ssh -N -R 54322:localhost:54322 upcloud_gpu
```

**Server, container shell:**

```bash
TELEOP_DEVICE=virtual PYTHONUNBUFFERED=1 \
/workspace/isaaclab/_isaac_sim/python.sh source/sim_to_real_so101/scripts/lerobot_agent.py \
  --task Lerobot-So101-Teleop-Vials-To-Rack-DR --teleop_device virtual --livestream 2 \
  --kit_args "--/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP> --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998" \
  2>&1 | tee outputs/virtual_teleop.log
```

The log prints `[VirtualLeader] connected`, and the page's status pill turns green. Keys: Q/A shoulder pan, W/S lift, E/D elbow, R/F wrist flex, T/G wrist roll, Y/H gripper, Shift for fine moves, Space to start/stop recording. The page's Record / Cancel / Reset buttons do the same as the agent's S / C / R keys, so they work even if keyboard input doesn't pass through the stream. Add `--repo_id`, `--repo_root` and `--task_name` to record.

Running the script file with `python.sh` instead of the `lerobot_agent` entry point guarantees the checked-out code runs. Order doesn't matter much: the agent keeps waiting for the app, and reconnects if it drops.

## 11. Logs and housekeeping

- Isaac Sim (Kit) logs: host `~/docker/isaac-sim/logs/Kit/<app>/5.1/kit_*.log`, container `/root/.nvidia-omniverse/logs/Kit/`.
- Isaac Lab logs go to `/tmp/isaaclab/logs` in the container and are lost with it. Pipe runs through `tee outputs/<name>.log` instead; `outputs/` is mounted from the host.
- Copy a log tail to the Mac clipboard: `ssh upcloud_gpu 'tail -n 80 ~/Sim-to-Real-SO-101-Workshop/outputs/agent_run.log' | pbcopy`
- Hung Isaac Sim after Ctrl-C or a crash: from another tmux window on the host, `docker exec teleop pkill -9 -f kit`, or `docker kill teleop` (the container uses `--rm`, so restart it with `remote/run_teleop.sh`). Then check that `nvidia-smi` no longer shows a `kit`/`python` process holding GPU memory.
- Code under `source/` and `remote/` is mounted into the container, so `git pull` on the host is enough; no image rebuild for code or asset changes.
- After a driver change, clear stale shader caches inside the container: `rm -rf /root/.cache/ov/shaders /root/.cache/nvidia/GLCache/*`.
- Done for the day: shut the server down in the Hub. Done with the course: delete the server **and** its storage volume.

## Troubleshooting quick reference

| Symptom | Cause / fix |
| --- | --- |
| `ssh: ... Operation timed out` | Home IP changed (update firewall), or server not started |
| `Permission denied (publickey)` | Use `User root`; check `IdentityFile` |
| Docker build: `curl: (22) ... 404` on ffmpeg | BtbN `latest` dropped 7.1; use a dated build (step 6) |
| `ldconfig -p | grep libnvidia-encode` empty in container | Missing `NVIDIA_DRIVER_CAPABILITIES=all`, or compute-only host driver (step 5) |
| Segfault in `librtx.scenedb.plugin.so` at startup | 595 driver; move to 580 (step 5) |
| `A camera was spawned without the --enable_cameras flag` | Add `--enable_cameras` |
| `Could not connect on port '/dev/ttyACM1'` | Leader arm not bridged (step 10), or use `preview_scene.py` |
| Black screen in client | App not fully loaded yet, wrong `publicIp`, or UDP blocked |
| `[VirtualLeader] waiting for teleop app on 127.0.0.1:54322` repeating | `joint_teleop_app.py` not running on the Mac, or the `ssh -R 54322` tunnel isn't up (step 10b) |
| Teleop page says "waiting for sim" | Agent not started with `--teleop_device virtual`, or tunnel down |
| Bottle renders black or grey | Translucency off; use the `Vials-Bottle` tasks, which enable it ([BOTTLE_SCENE.md](BOTTLE_SCENE.md)) |
| `Encountered 1 file that should have been a pointer` | An LFS-tracked file was committed as a normal blob; `git add --renormalize <file>` and commit |
| USD or texture loads fail, file is ~130 bytes | LFS pointer stub; run `git lfs pull` |
