# Adding a transparent bottle to the vials scene

This fork adds a small transparent PET bottle as a new prop and two new tasks that put it into the course's vials-to-rack scene. The course's own tasks are unchanged. Nothing about the bottle is wired into rewards, success checks or randomization yet; it is just a rigid object sitting on the mat next to the vials and the rack.

It also adds a way to teleoperate the sim without a leader arm (joint sliders in a browser on the Mac), which is the easiest way to poke at the bottle. That part works with every task, not only the bottle ones.

Server-side setup (container, streaming, tunnels) is in [REMOTE_HOST_SETUP.md](REMOTE_HOST_SETUP.md).

## What was added

| File | What it is |
| --- | --- |
| `asset_gen/make_bottle.py` | Generates the bottle USD procedurally from measured dimensions. Runs anywhere with `usd-core`; no Isaac needed. |
| `asset_gen/requirements.txt` | Python deps for the generator. |
| `source/sim_to_real_so101/utils/bottle_fill.py` | Optional bead or PBD liquid fill for the bottle. |
| `source/sim_to_real_so101/assets/usd/Bottle.usda` | The generated asset used by the tasks. |
| `source/sim_to_real_so101/tasks/vials_bottle_env_cfg.py` | Scene, event and env configs for the new tasks. |
| `source/sim_to_real_so101/tasks/__init__.py` | Two `gym.register` calls appended at the end. |
| `remote/joint_teleop_app.py` | Joint-slider teleop UI, runs on the Mac. Standard library only. |
| `source/sim_to_real_so101/utils/virtual_leader.py` | Sim-side stand-in for the leader arm that the slider app drives. |
| `source/sim_to_real_so101/scripts/lerobot_agent.py` | New `--teleop_device virtual` option. Default is still the physical leader. |

New tasks:

- `Lerobot-So101-Teleop-Vials-Bottle`: vials-to-rack scene plus the bottle.
- `Lerobot-So101-Teleop-Vials-Bottle-DR`: the same with the course's domain randomization (HDRI, lights, robot colour, mat, cameras).

## The bottle asset

The bottle is a solid of revolution built from a 2D profile: rounded base, straight cylinder body, rounded shoulder, threaded neck, open top. Measured from the real bottle:

| Part | Size |
| --- | --- |
| Body diameter | 39 mm |
| Straight body height | 90 mm |
| Shoulder height | 10 mm (tangent quarter-ellipse; `--shoulder sphere` gives the literal ball surface, which leaves a crease at the body) |
| Neck diameter | 20 mm, over the thread crests |
| Neck height | 16 mm |
| Total height | 116 mm, capacity about 112 ml |

The checked-in `Bottle.usda` was generated with `--shoulder ellipse --mass-g 12 --wall 0.3`. The exact parameters are stored in the root prim's `comment` metadata, so you can always tell what a given file was built with.

What's inside the USD (Z-up, metres, origin at the centre of the base):

- `/Bottle`: the rigid body. `RigidBodyAPI` plus `MassAPI` with mass, centre of mass and the diagonal inertia of the actual thin shell, not of a solid cylinder.
- `/Bottle/visuals`: the render mesh, hollow, with real wall thickness (0.3 mm body, 1.2 mm neck, 1.5 mm base) and three ring ridges approximating the thread. Bound to an OmniGlass MDL material (IOR 1.57 for PET) with a `UsdPreviewSurface` fallback, so it also shows up in macOS Quick Look and other viewers.
- `/Bottle/collision`: invisible colliders with no threads and a physics material with static/dynamic friction 0.5/0.4. The render mesh does not collide. The outside matches the bottle in both collider modes, so grasping behaves the same.
  - `--collider hollow` (default): a compound of about 110 convex pieces (floor plus wall bands × 16 sectors), with an open interior so beads or liquid can sit inside. The walls are 2.5 mm thick (`--collider-wall`), thicker than the real 0.3 mm, so small objects can't tunnel through. The usable interior is about 88 ml instead of 115 ml. The interior profile is stored in the root prim's customData (`interior_r_m`, `interior_z_m`) for the fill code.
  - `--collider hull`: a single solid convex hull. Cheaper, but nothing fits inside.

### Regenerating it

On the Mac, in the repo root:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate   # or: python3.12 -m venv .venv
uv pip install -r asset_gen/requirements.txt               # or: pip install -r ...

python asset_gen/make_bottle.py --shoulder ellipse --mass-g 12 --wall 0.3 \
  -o source/sim_to_real_so101/assets/usd/Bottle.usda --preview outputs/bottle.png
open outputs/bottle.png                                     # cross-section with dimensions
qlmanage -p source/sim_to_real_so101/assets/usd/Bottle.usda # rotatable 3D view (preview material)
```

`python asset_gen/make_bottle.py --help` lists every parameter: dimensions, wall thicknesses, fillet, thread depth and count, mass, friction, IOR, frosting, mesh resolution. `trimesh` and `scipy` are only for the inertia and watertight check; without them the USD is still written, minus explicit inertia. Quick Look can't render the MDL glass, so judge the real look in Isaac Sim.

`*.usda` is tracked by Git LFS in this repo, so run `git lfs install` once on the Mac before committing a regenerated file. If git warns `Encountered 1 file that should have been a pointer`, the file went in as a normal blob; fix it with:

```bash
git add --renormalize source/sim_to_real_so101/assets/usd/Bottle.usda
git commit -m "Store Bottle.usda in LFS"
```

## The tasks

`vials_bottle_env_cfg.py` subclasses the existing vials-to-rack configs and adds only three things:

1. **`bottle`**, a `RigidObjectCfg` pointing at `Bottle.usda`. It spawns with its base at (0.12, -0.12, 0.045), between the robot and the vials, about 1 cm above the mat so it drops and settles. Mass, collider and friction come from the USD; the config only adds some angular damping. To move it, change `pos` in `BOTTLE_CFG`.
2. **`reset_bottle`**, an event using Isaac Lab's `reset_root_state_uniform` with empty ranges. It puts the bottle back at its start pose on every reset. Give it `pose_range` values to randomize its position.
3. **`self.sim.render.enable_translucency = True`**. The base config turns RTX translucency off; without it the glass renders dark or opaque. This only applies to the new tasks.

Success checks, observations and the vial and rack randomization are inherited unchanged.

## Running it

All server commands run inside the teleop container (`remote/run_teleop.sh`), from `/workspace/Sim-to-Real-SO-101-Workshop`. To keep the commands short, set the stream flags once:

```bash
export KIT="--/exts/omni.kit.livestream.app/primaryStream/publicIp=<SERVER_IP> --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998"
```

### Just look at it

```bash
/workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py \
  --task Lerobot-So101-Teleop-Vials-Bottle-DR --livestream 2 --enable_cameras --kit_args "$KIT" \
  2>&1 | tee outputs/bottle_preview.log
```

Connect the streaming client. The bottle should drop onto the mat, stand still within a second and look clear rather than grey or black. The scene resets every 15 s.

### Teleop with the joint sliders (no leader arm)

Mac, two terminals:

```bash
python3 remote/joint_teleop_app.py              # then open http://127.0.0.1:8765
ssh -N -R 54322:localhost:54322 upcloud_gpu
```

Server, in the container:

```bash
TELEOP_DEVICE=virtual PYTHONUNBUFFERED=1 \
/workspace/isaaclab/_isaac_sim/python.sh source/sim_to_real_so101/scripts/lerobot_agent.py \
  --task Lerobot-So101-Teleop-Vials-Bottle-DR --teleop_device virtual \
  --livestream 2 --kit_args "$KIT" \
  2>&1 | tee outputs/virtual_teleop.log
```

The agent log should print `[VirtualLeader] connected`, and the page's status pill turns green. In the page:

- **Sliders**: one per joint, in leader units (arm -100..100, gripper 0..100) with approximate degrees.
- **Keys** (hold to move): Q/A shoulder pan, W/S shoulder lift, E/D elbow, R/F wrist flex, T/G wrist roll, Y/H gripper. Shift for fine moves; the speed slider sets the rate.
- **Buttons**: Record (Space) / Stop & save, Cancel episode, Home, Reset world. These do the same as the agent's S / C / R keys, so you don't depend on keyboard input passing through the stream.

The sim side rate-limits each joint (about 120 units/s), so jumping a slider moves the arm quickly but doesn't snap it. On reset, the sliders return to the start pose.

Running the script file through `python.sh`, rather than the `lerobot_agent` entry point, makes sure the checked-out code is what runs. `lerobot_agent --teleop_device virtual ...` works too once the package is installed editable.

### Teleop with the physical leader arm

Same as section 10 of [REMOTE_HOST_SETUP.md](REMOTE_HOST_SETUP.md), just with `--task Lerobot-So101-Teleop-Vials-Bottle-DR`.

### Recording demos

Add the course's dataset args to either teleop command:

```bash
  --repo_id <hf_user>/<dataset_name> --repo_root datasets/<dataset_name> --task_name "<instruction>"
```

Datasets recorded with the sliders are in the same LeRobot format as leader-arm ones: actions in leader units and the same camera streams. Expect slider demos to be slower and less smooth than leader-arm demos, though, and that matters for policy quality.

## Filling the bottle: beads or PBD liquid

The bottle can optionally be filled, still in the same `Vials-Bottle` tasks. You choose the fill with environment variables at launch, and nothing changes when they're unset. The code is in `source/sim_to_real_so101/utils/bottle_fill.py`; the switch is `apply_bottle_fill()` in `vials_bottle_env_cfg.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `BOTTLE_FILL` | `none` | `none`, `beads` or `pbd` |
| `BOTTLE_FILL_FRACTION` | `0.4` | liquid level as a fraction of the inside height (floor to rim); 0.4 ≈ 48 mm ≈ 41 ml |
| `BOTTLE_BEAD_RADIUS_MM` | `2.5` | beads: sphere radius |
| `BOTTLE_MAX_BEADS` | `600` | beads: cap on rigid bodies (the level drops if hit) |
| `BOTTLE_PBD_SPACING_MM` | `2.5` | pbd: particle spacing; halving it means 8× the particles |
| `BOTTLE_PBD_ISOSURFACE` | `0` | pbd: `1` renders a continuous liquid surface instead of particle spheres |
| `BOTTLE_PBD_LOOK` | `blue` | pbd: `blue` = opaque blue, easy to see; `water` = clear, IOR 1.333 (realistic, but nearly invisible inside a clear bottle) |
| `BOTTLE_PBD_DEBUG` | `0` | pbd: `1` prints particle count, height range, how far they moved and how many are inside the bottle, every 2 s (needs `--disable_fabric`) |

In both modes, the fill gets the mass of water at that level (41 g at 0.4, versus the 12 g bottle). It's packed on a staggered lattice inside the hollow collider at the bottle's start pose, so it drops onto the mat together with the bottle.

Needs `Bottle.usda` generated with the hollow collider (the generator's default since this change). An old file fails with a message telling you to regenerate.

### Beads

About 180 small blue spheres at fill 0.4 (2.5 mm radius, ~0.23 g each, almost frictionless, no bounce), as an Isaac Lab `RigidObjectCollection` named `beads`. They're ordinary rigid bodies, so they work with everything, and a reset puts them back inside the bottle along with it. Use beads for shifting mass, tipping and rough pouring.

```bash
BOTTLE_FILL=beads BOTTLE_FILL_FRACTION=0.4 \
/workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py \
  --task Lerobot-So101-Teleop-Vials-Bottle-DR --livestream 2 --enable_cameras --kit_args "$KIT" \
  2>&1 | tee outputs/bottle_beads.log
```

The log prints `[bottle_fill] beads: N x r=2.5 mm, level .. mm, total .. g`. Startup takes a few seconds longer, since each bead is spawned as its own prim. The same variables work with the teleop command, so you can push or grasp the bottle with the sliders and watch the beads shift.

Or teleoperate:

````bash
BOTTLE_FILL=beads BOTTLE_FILL_FRACTION=0.4 TELEOP_DEVICE=virtual PYTHONUNBUFFERED=1 \
/workspace/isaaclab/_isaac_sim/python.sh source/sim_to_real_so101/scripts/lerobot_agent.py \
  --task Lerobot-So101-Teleop-Vials-Bottle-DR --teleop_device virtual \
  --livestream 2 --kit_args "$KIT" \
  2>&1 | tee outputs/teleop_beads.log
```

### PBD liquid (experimental)

A PhysX position-based-dynamics fluid: about 1,800 particles at fill 0.4 and 2.5 mm spacing, with water-like viscosity, cohesion and surface tension, rendered with an IOR 1.333 glass material. Isaac Lab has no asset class for particles, so a `prestartup` event creates the particle system and particle set on the USD stage before the sim starts. That requires `replicate_physics = False`, which the config sets automatically; teleop uses one env anyway.

```bash
BOTTLE_FILL=pbd BOTTLE_FILL_FRACTION=0.4 BOTTLE_PBD_ISOSURFACE=1 \
/workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py \
  --task Lerobot-So101-Teleop-Vials-Bottle-DR --livestream 2 --enable_cameras --disable_fabric --kit_args "$KIT" \
  2>&1 | tee outputs/bottle_pbd.log
```

Things to expect and check:

- **Start without the isosurface** (`BOTTLE_PBD_ISOSURFACE=0`). Particles then render as small spheres, which makes leaks and jitter obvious. Switch the isosurface on once the physics looks right; it costs extra GPU time every frame.
- **`--disable_fabric`** is recommended. Particles have known rough edges with Fabric on Isaac Sim 5.x, and the reset writes particle positions back through USD. If the liquid doesn't return to the bottle on reset, the reset path isn't reaching PhysX in your build; restart the run to refill. `lerobot_agent.py` takes `--disable_fabric` too.
- **Leaks or particles at rest jittering** mean the spacing is too coarse relative to the 2.5 mm walls, or there are too few solver iterations. Try `BOTTLE_PBD_SPACING_MM=2.0`, or regenerate the bottle with `--collider-wall 3`.
- **Performance.** Watch the stream's frame rate and `nvidia-smi`. A few thousand particles should be fine on an L40S. The isosurface and smaller spacing are what get expensive.
- **The particle API is set up through `PhysxSchema` directly.** If a schema attribute name differs in your Isaac Sim build, the traceback points at `spawn_pbd_liquid` in `bottle_fill.py`; paste it and it's a quick fix.

## Tumbler glass (asset only, not in a task yet)

`asset_gen/make_tumbler.py` builds a cut-crystal tumbler modelled on the Riedel DSG Neat, with the same conventions as the bottle: Z-up, metres, origin at the centre of the underside, glass material, compound collider with an open interior, and the interior profile in customData so `utils/bottle_fill.py` can fill it. Defaults are the measured glass:

| Part | Size |
| --- | --- |
| Height / outer diameter | 82 / 82 mm |
| Solid base (inside floor height) | 13 mm |
| Wall | 6 mm at the floor, tapering to 2 mm at the top, fully rounded rim |
| Cut pattern | 15 V-cut dents: an 11 mm wide, 2.5 mm deep triangular notch in the bottom edge, narrowing up to 28 mm with nearly straight sides that then curve in (`--hull-h`, `--hull-a`, `--hull-b`), then a straight 2 mm groove (`--groove-width`, `--groove-depth`) that narrows and fades out 3 mm below the rim. `--no-cuts` gives a plain glass |
| Underside | Concave, cosine-bell profile 3 mm higher at the centre (`--punt-depth`), resting on a ring at the edge; a radial V-cut from each side notch narrowing to a point at a smooth Ø10 mm centre disc; side and bottom cut form one groove wrapping smoothly around the bottom edge (`--centre-d`, `--no-bottom-cuts`) |
| Mass | ~307 g, from the mesh volume × 2.45 g/cm³ crystal (`--mass-g` to use a weighed value) |
| Capacity | ~297 ml brim-full |

```bash
python asset_gen/make_tumbler.py -o source/sim_to_real_so101/assets/usd/Tumbler.usda --preview outputs/tumbler.png
```

Riedel's product page lists the Neat as 77 mm high × 70 mm wide, so measure again before relying on the defaults (`--height`, `--diameter`). The render mesh is ~62k faces (6.5 MB); `--res-mm` (circumferential spacing, 4× finer around each groove) and `--dz` trade detail for size. The preview shows the section, the outside unrolled with the cut depth (the quickest way to judge the dent outline), the outside, and the underside seen from below. The collider ignores the cuts: a solid base disc plus 16 convex wall sectors following the inner taper.

## Glassware task: mat, bottle and tumbler only

`Lerobot-So101-Teleop-Glassware` (and `-DR`) is the course's base task scene (robot, lightbox with its light and D455 camera, wrist camera, mat) with only the bottle and the tumbler on the mat. There are no vials, no rack and no success terms. It's defined in `source/sim_to_real_so101/tasks/glassware_env_cfg.py`, and the other tasks are untouched.

| Prop | Start position (x, y, z), m | Reset |
| --- | --- | --- |
| Bottle | (0.22, -0.10, 0.045) | back to start pose |
| Tumbler | (0.20, 0.09, 0.037) | back to start pose |

Both drop a few mm onto the mat at start. To move them, change `GLASS_BOTTLE_POS` and `TUMBLER_POS`. The base task's light, mat and camera randomization applies. `-DR` adds the same robot colour, HDRI sky and wider mat rotation randomization as the vials DR task. Translucency and the refraction settings come from the base task config.

Both assets have to exist first:

```bash
python asset_gen/make_bottle.py --shoulder ellipse --mass-g 12 --wall 0.3 -o source/sim_to_real_so101/assets/usd/Bottle.usda
python asset_gen/make_tumbler.py -o source/sim_to_real_so101/assets/usd/Tumbler.usda
```

Look at it, or teleop it with the leader arm (REMOTE_HOST_SETUP.md step 10c) or the sliders:

```bash
/workspace/isaaclab/_isaac_sim/python.sh remote/preview_scene.py \
  --task Lerobot-So101-Teleop-Glassware-DR --livestream 2 --enable_cameras --kit_args "$KIT" \
  2>&1 | tee outputs/glassware_preview.log

TELEOP_DEVICE=virtual PYTHONUNBUFFERED=1 \
/workspace/isaaclab/_isaac_sim/python.sh source/sim_to_real_so101/scripts/lerobot_agent.py \
  --task Lerobot-So101-Teleop-Glassware-DR --teleop_device virtual --livestream 2 --kit_args "$KIT" \
  2>&1 | tee outputs/glassware_teleop.log
```

The bead and liquid fills work here too, in either glass: `BOTTLE_FILL=beads` (or `pbd`) as above, plus `FILL_TARGET=bottle` (default) or `FILL_TARGET=tumbler`.

## Iterating

1. Change parameters, regenerate `Bottle.usda` on the Mac, check the PNG and Quick Look.
2. Commit (LFS) and push; `git pull` on the server.
3. Rerun the preview or the teleop command. No container rebuild is needed: `source/` and `remote/` are mounted.

## Known limits and next steps

- **Transparency is the hard part for sim-to-real.** The D455 depth is unreliable on clear plastic, and rendered PET won't look exactly like the real bottle under your lights. Candidates:
  - randomize glass tint, frosting and IOR per reset;
  - add an opaque cap or label;
  - record some episodes with an opaque variant.
- **Mass and friction** are estimates (12 g; 0.5/0.4). Weigh the real bottle; tune friction if grasps slip.
- **Gripper fit.** The 39 mm body may be close to the SO-101 jaw's maximum opening; the 20 mm neck is the easy grasp.
- **Placement** was chosen from scene coordinates. Adjust `pos` if the bottle blocks a camera's view of the vials.
- **No task semantics yet.** The bottle isn't part of any observation, reward or success term. A bottle task would add those in a new env cfg, the same way the vials task does.
- **The fill is a stand-in.** Beads get mass and sloshing roughly right but not surface tension. PBD looks like liquid but isn't calibrated to water's flow. Neither is needed for pick-and-place; use them where the liquid's motion actually matters, such as pouring or fast moves.
