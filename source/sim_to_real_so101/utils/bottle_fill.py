"""Fill the bottle prop with rigid beads or PBD liquid particles.

Both modes need Bottle.usda generated with `--collider hollow` (the default): its root prim
carries the collider's interior profile (customData interior_r_m / interior_z_m), which is
used here to pack beads or particles inside the bottle at its start pose.

beads : a RigidObjectCollection of small spheres. Plain rigid bodies, so they work with
        everything in Isaac Lab, and they're reset with the bottle.
pbd   : a PhysX position-based-dynamics fluid (particle system + particle set), created on
        the USD stage before the simulation starts. Isaac Lab has no asset class for it, so
        it's set up with PhysxSchema directly. Experimental: see BOTTLE_SCENE.md.
"""
from __future__ import annotations

import math

import numpy as np

WATER_DENSITY = 1000.0  # kg/m^3


# ----------------------------------------------------------------------------- geometry


def read_interior(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Interior profile (r, z) in metres, bottle-local, floor to rim."""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    data = root.GetCustomData() if root else {}
    if "interior_r_m" not in data:
        raise RuntimeError(
            f"{usd_path} has no interior profile. Regenerate it with asset_gen/make_bottle.py "
            "--collider hollow (the default) to use BOTTLE_FILL=beads|pbd."
        )
    r = np.array(data["interior_r_m"], dtype=float)
    z = np.array(data["interior_z_m"], dtype=float)
    order = np.argsort(z, kind="stable")
    return r[order], z[order]


def _radius_limit(r, z, zq, half_band):
    """Smallest interior radius within [zq - half_band, zq + half_band]."""
    zs = np.linspace(zq - half_band, zq + half_band, 7)
    return float(np.min(np.interp(zs, z, r, left=0.0, right=0.0)))


def fill_level(z: np.ndarray, fraction: float) -> float:
    """Liquid surface height for a fill fraction of the inside height (floor to rim)."""
    return float(z[0] + np.clip(fraction, 0.0, 1.0) * (z[-1] - z[0]))


def liquid_volume(r: np.ndarray, z: np.ndarray, level: float) -> float:
    """Volume (m^3) of the interior below `level`."""
    zs = np.linspace(z[0], level, 400)
    rs = np.interp(zs, z, r)
    return float(np.trapezoid(math.pi * rs**2, zs)) if hasattr(np, "trapezoid") else \
        float(np.trapz(math.pi * rs**2, zs))


def pack_positions(r, z, spacing: float, level: float, clearance: float) -> np.ndarray:
    """Points on a staggered cubic lattice inside the interior, below `level`.

    clearance: minimum distance from a point to the walls and floor (e.g. bead radius + margin).
    Returns (N, 3) bottle-local positions in metres.
    """
    pts = []
    zc = z[0] + clearance
    layer = 0
    while zc <= level - spacing / 2 + 1e-9:
        rmax = _radius_limit(r, z, zc, clearance) - clearance
        if rmax > 0:
            off = (spacing / 2) * (layer % 2)
            n = int(rmax / spacing) + 2
            g = (np.arange(-n, n + 1) * spacing) + off
            xx, yy = np.meshgrid(g, g)
            m = np.hypot(xx, yy) <= rmax
            pts += [(x, y, zc) for x, y in zip(xx[m], yy[m])]
        zc += spacing
        layer += 1
    return np.array(pts, dtype=float).reshape(-1, 3)


# ----------------------------------------------------------------------------- beads


def make_bead_collection_cfg(usd_path: str, bottle_pos, fill: float, bead_radius: float,
                             max_beads: int = 600, mass_scale: float = 1.0):
    """RigidObjectCollectionCfg with beads packed inside the bottle at `bottle_pos` (env-local).

    Total bead mass = mass of water filling the bottle to the same level (x mass_scale).
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    r, z = read_interior(usd_path)
    level = fill_level(z, fill)
    local = pack_positions(r, z, spacing=2.04 * bead_radius, level=level, clearance=1.15 * bead_radius)
    if len(local) == 0:
        raise RuntimeError("No beads fit: increase BOTTLE_FILL_FRACTION or reduce the bead radius.")
    if len(local) > max_beads:
        local = local[:max_beads]  # lowest layers first, so this just lowers the level
        level = float(local[:, 2].max() + bead_radius)
        print(f"[bottle_fill] capped at {max_beads} beads (raise BOTTLE_MAX_BEADS for more)")
    total_mass = liquid_volume(r, z, level) * WATER_DENSITY * mass_scale
    m = total_mass / len(local)
    print(f"[bottle_fill] beads: {len(local)} x r={bead_radius * 1e3:.1f} mm, "
          f"level {level * 1e3:.0f} mm, total {total_mass * 1e3:.0f} g", flush=True)

    spawn = sim_utils.SphereCfg(
        radius=bead_radius,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, solver_position_iteration_count=8, linear_damping=0.1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=m),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.4 * bead_radius, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.05, dynamic_friction=0.05, restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.95), roughness=0.3),
    )
    bx, by, bz = bottle_pos
    objects = {
        f"bead_{i:03d}": RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Bead_{i:03d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(bx + x, by + y, bz + zz)),
        )
        for i, (x, y, zz) in enumerate(local)
    }
    return RigidObjectCollectionCfg(rigid_objects=objects)


def reset_beads(env, env_ids, asset_cfg):
    """Event: put every bead back at its packed start position inside the bottle."""
    import torch

    beads = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    state = beads.data.default_object_state[env_ids].clone()
    state[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    beads.write_object_state_to_sim(state, env_ids=env_ids)


# ----------------------------------------------------------------------------- PBD liquid

_PBD_SETS: dict[str, np.ndarray] = {}  # particle set prim path -> initial world positions


def _apply_api(prim, *names):
    """Apply whichever of the named PhysxSchema API classes exist; return the applied ones."""
    from pxr import PhysxSchema

    out = []
    for n in names:
        cls = getattr(PhysxSchema, n, None)
        if cls is not None:
            out.append(cls.Apply(prim))
    return out


def _set_attr(prim, apis, name, attr_name, sdf_type, value):
    """Set an attribute via the first schema object with Create<name>Attr, else by raw attribute name.

    PhysX schema attributes move between API classes across Isaac Sim versions; writing the
    raw attribute (e.g. physxParticle:selfCollision) is equivalent and doesn't depend on that.
    """
    for api in apis:
        fn = getattr(api, f"Create{name}Attr", None)
        if fn is not None:
            return fn(value)
    attr = prim.CreateAttribute(attr_name, sdf_type)
    attr.Set(value)
    return attr


def _set_rel(prim, apis, name, rel_name, targets):
    for api in apis:
        fn = getattr(api, f"Create{name}Rel", None)
        if fn is not None:
            return fn().SetTargets(targets)
    return prim.CreateRelationship(rel_name).SetTargets(targets)


def spawn_pbd_liquid(env, env_ids, usd_path: str, bottle_pos, fill: float, spacing: float,
                     isosurface: bool = False, look: str = "blue", viscosity: float = 0.0091,
                     cohesion: float = 0.01, surface_tension: float = 0.0074, friction: float = 0.1):
    """Prestartup event: create a PBD particle system and one fluid particle set per env.

    Offsets follow the PhysX fluid convention: fluid rest offset = spacing / 2, and
    particle contact offset = fluid rest offset / 0.6.

    look: "blue" = opaque blue (easy to see, good for checking the physics);
          "water" = clear glass with IOR 1.333 (realistic, but nearly invisible inside a
          clear bottle, especially as separate particles without the isosurface).
    """
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    scene_path = env.sim.cfg.physics_prim_path
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path)).CreateEnableGPUDynamicsAttr(True)

    fluid_rest = spacing / 2
    particle_contact = fluid_rest / 0.6
    solid_rest = particle_contact * 0.99
    contact = particle_contact

    psys_path = "/World/Liquid/ParticleSystem"
    UsdGeom.Xform.Define(stage, "/World/Liquid")
    psys = PhysxSchema.PhysxParticleSystem.Define(stage, psys_path)
    sp, F, I, B = psys.GetPrim(), Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Int, Sdf.ValueTypeNames.Bool
    _set_rel(sp, [psys], "SimulationOwner", "simulationOwner", [Sdf.Path(scene_path)])
    _set_attr(sp, [psys], "ContactOffset", "contactOffset", F, contact)
    _set_attr(sp, [psys], "RestOffset", "restOffset", F, solid_rest)
    _set_attr(sp, [psys], "ParticleContactOffset", "particleContactOffset", F, particle_contact)
    _set_attr(sp, [psys], "SolidRestOffset", "solidRestOffset", F, solid_rest)
    _set_attr(sp, [psys], "FluidRestOffset", "fluidRestOffset", F, fluid_rest)
    _set_attr(sp, [psys], "SolverPositionIterationCount", "solverPositionIterationCount", I, 16)
    _set_attr(sp, [psys], "MaxVelocity", "maxVelocity", F, 2.0)

    # fluid behaviour
    pbd = UsdShade.Material.Define(stage, "/World/Liquid/PBDMaterial")
    mp = pbd.GetPrim()
    pm = _apply_api(mp, "PhysxPBDMaterialAPI")
    _set_attr(mp, pm, "Viscosity", "physxPBDMaterial:viscosity", F, viscosity)
    _set_attr(mp, pm, "Cohesion", "physxPBDMaterial:cohesion", F, cohesion)
    _set_attr(mp, pm, "SurfaceTension", "physxPBDMaterial:surfaceTension", F, surface_tension)
    _set_attr(mp, pm, "Friction", "physxPBDMaterial:friction", F, friction)
    _set_attr(mp, pm, "Density", "physxPBDMaterial:density", F, WATER_DENSITY)
    UsdShade.MaterialBindingAPI.Apply(psys.GetPrim()).Bind(
        pbd, UsdShade.Tokens.weakerThanDescendants, "physics")

    # look: clear water (IOR 1.333)
    water = UsdShade.Material.Define(stage, "/World/Liquid/Water")
    sh = UsdShade.Shader.Define(stage, "/World/Liquid/Water/OmniGlass")
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.93, 1.0))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.333)
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    water.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")

    blue = UsdShade.Material.Define(stage, "/World/Liquid/Blue")
    bs = UsdShade.Shader.Define(stage, "/World/Liquid/Blue/PreviewSurface")
    bs.CreateIdAttr("UsdPreviewSurface")
    bs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.35, 0.95))
    bs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    blue.CreateSurfaceOutput().ConnectToSource(bs.ConnectableAPI(), "surface")
    liquid_mat = water if look == "water" else blue

    if isosurface:
        try:
            iso = _apply_api(sp, "PhysxParticleIsosurfaceAPI")
            if not iso:
                raise RuntimeError("PhysxParticleIsosurfaceAPI not in this PhysxSchema")
            S = Sdf.ValueTypeNames.String
            for n, t, v in [("IsosurfaceEnabled", B, True), ("MaxVertices", I, 1024 * 1024),
                            ("MaxTriangles", I, 2 * 1024 * 1024), ("MaxSubgrids", I, 1024 * 4),
                            ("GridSpacing", F, fluid_rest * 1.5), ("SurfaceDistance", F, fluid_rest * 1.6),
                            ("GridFilteringPasses", S, ""), ("GridSmoothingRadius", F, fluid_rest * 2),
                            ("NumMeshSmoothingPasses", I, 4), ("NumMeshNormalSmoothingPasses", I, 4)]:
                _set_attr(sp, iso, n, "physxIsosurface:" + n[0].lower() + n[1:], t, v)
            UsdShade.MaterialBindingAPI.Apply(psys.GetPrim()).Bind(liquid_mat)
        except Exception as e:  # noqa: BLE001
            print(f"[bottle_fill] isosurface not available ({e}); rendering particles as spheres")
            isosurface = False

    r, z = read_interior(usd_path)
    level = fill_level(z, fill)
    local = pack_positions(r, z, spacing=spacing, level=level, clearance=solid_rest + 0.25 * spacing)
    total_mass = liquid_volume(r, z, level) * WATER_DENSITY
    origins = env.scene.env_origins.detach().cpu().numpy()
    ids = range(env.scene.num_envs) if env_ids is None else [int(i) for i in env_ids]
    for i in ids:
        world = local + np.asarray(bottle_pos) + origins[i]
        path = f"/World/Liquid/Particles_env{i}"
        pts = UsdGeom.Points.Define(stage, path)
        pts.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in world]))
        pts.CreateVelocitiesAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * len(world)))
        pts.CreateWidthsAttr(Vt.FloatArray([2 * fluid_rest] * len(world)))
        pp = pts.GetPrim()
        # selfCollision / particleGroup / particleSystem belong to PhysxParticleAPI, fluid to
        # PhysxParticleSetAPI; apply both and fall back to the raw physxParticle:* names
        ps = _apply_api(pp, "PhysxParticleSetAPI", "PhysxParticleAPI")
        _set_attr(pp, ps, "SelfCollision", "physxParticle:selfCollision", B, True)
        _set_attr(pp, ps, "Fluid", "physxParticle:fluid", B, True)
        _set_attr(pp, ps, "ParticleGroup", "physxParticle:particleGroup", I, 0)
        _set_rel(pp, ps, "ParticleSystem", "physxParticle:particleSystem", [Sdf.Path(psys_path)])
        UsdPhysics.MassAPI.Apply(pts.GetPrim()).CreateMassAttr(total_mass)
        if isosurface:
            pts.CreateVisibilityAttr(UsdGeom.Tokens.invisible)  # the isosurface mesh is what renders
        else:
            UsdShade.MaterialBindingAPI.Apply(pts.GetPrim()).Bind(liquid_mat)
        _PBD_SETS[path] = world
    print(f"[bottle_fill] PBD liquid: {len(local)} particles/env, spacing {spacing * 1e3:.1f} mm, "
          f"level {level * 1e3:.0f} mm, {total_mass * 1e3:.0f} g, isosurface={'on' if isosurface else 'off'}, "
          f"look={look}, prims under /World/Liquid", flush=True)


def reset_pbd_liquid(env, env_ids):
    """Reset event: write the initial particle positions back (USD write; best with --disable_fabric)."""
    import omni.usd
    from pxr import Gf, UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    for path, world in _PBD_SETS.items():
        pts = UsdGeom.Points(stage.GetPrimAtPath(path))
        if not pts:
            continue
        pts.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in world]))
        pts.GetVelocitiesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0)] * len(world)))


def report_pbd_liquid(env, env_ids, bottle_cfg=None):
    """Interval event (BOTTLE_PBD_DEBUG=1): print where the particles are.

    Reads the particle positions PhysX writes back to USD, which only happens with
    --disable_fabric. If the numbers never change, either Fabric is on or the particles
    aren't being simulated.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    bottle_xy = None
    if bottle_cfg is not None:
        try:
            pos = env.scene[bottle_cfg.name].data.root_pos_w[0].detach().cpu().numpy()
            bottle_xy = pos[:2]
        except Exception:  # noqa: BLE001
            pass
    for path, start in _PBD_SETS.items():
        pts = UsdGeom.Points(stage.GetPrimAtPath(path))
        if not pts:
            print(f"[bottle_fill] {path} missing from the stage", flush=True)
            continue
        p = np.array(pts.GetPointsAttr().Get(), dtype=float)
        moved = float(np.max(np.linalg.norm(p - start, axis=1))) if len(p) == len(start) else float("nan")
        msg = (f"[bottle_fill] {path}: n={len(p)} z min/mean/max = {p[:, 2].min() * 1e3:.0f}/"
               f"{p[:, 2].mean() * 1e3:.0f}/{p[:, 2].max() * 1e3:.0f} mm, max moved {moved * 1e3:.1f} mm")
        if bottle_xy is not None:
            inside = np.mean(np.linalg.norm(p[:, :2] - bottle_xy, axis=1) < 0.0195)
            msg += f", {inside * 100:.0f}% within the bottle footprint"
        print(msg, flush=True)
