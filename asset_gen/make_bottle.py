#!/usr/bin/env python3
"""Procedurally build a small transparent PET bottle as a sim-ready USD asset.

The bottle is a solid of revolution generated from a 2D profile:
  base (with fillet) -> straight cylinder body -> spherical "ball" shoulder -> threaded neck (open top)

Output USD (Z-up, metres, origin at the centre of the bottle's base):
  /Bottle                 Xform, defaultPrim, RigidBodyAPI + MassAPI (mass, COM, shell inertia)
  /Bottle/visuals         thin-walled render mesh (outer + inner wall), glass material
  /Bottle/collision       invisible colliders with plastic friction. --collider hollow (default):
                          compound of convex wall pieces with an open interior, so beads or PBD
                          liquid can sit inside. --collider hull: one solid convex hull (cheaper).
  /Bottle/Looks/...       OmniGlass MDL (Isaac Sim) with UsdPreviewSurface fallback, physics material

Needs `pip install usd-core numpy`; scipy for the hollow collider, trimesh for inertia + checks.
With --collider hollow the root prim's customData also carries the collider's interior profile
(interior_r_m / interior_z_m), which the sim uses to place beads or liquid particles inside.
Run anywhere (Mac, server, inside the container with /isaac-sim/python.sh).

  python3 make_bottle.py -o Bottle.usda
  python3 make_bottle.py --body-d 39 --body-h 90 --shoulder-h 10 --neck-d 20 --neck-h 16 --mass-g 15
  python3 make_bottle.py --shoulder ellipse --no-threads --preview bottle_preview.png
  python3 make_bottle.py --shoulder ellipse --spout -o BottleSpout.usda   # with a pour spout
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

MM = 1e-3


@dataclass
class BottleParams:
    body_d: float = 39.0        # outer diameter of the cylindrical body, mm
    body_h: float = 90.0        # height of the straight cylinder (from the floor), mm
    shoulder_h: float = 10.0    # height of the rounded shoulder, mm
    neck_d: float = 20.0        # outer diameter of the neck (over the thread crests), mm
    neck_h: float = 16.0        # height of the neck, mm
    wall: float = 0.3           # body/shoulder wall thickness, mm (thin PET)
    neck_wall: float = 1.2      # neck finish wall thickness, mm (measured at thread root)
    base_wall: float = 1.5      # bottom thickness, mm
    base_fillet: float = 2.5    # radius of the rounded edge at the base, mm
    shoulder: str = "sphere"    # "sphere" (ball surface through both ends) or "ellipse" (tangent to body)
    threads: bool = True        # add visual thread ridges on the neck
    thread_depth: float = 0.8   # radial depth of the thread ridges, mm
    thread_count: int = 3       # number of ridges (rings approximating the helix)
    mass_g: float = 15.0        # empty bottle mass, grams (weigh yours!)
    static_friction: float = 0.5
    dynamic_friction: float = 0.4
    restitution: float = 0.1
    ior: float = 1.57           # PET
    frosting: float = 0.02      # 0 = perfectly clear
    segments: int = 96          # angular resolution of the render mesh
    collision_segments: int = 32
    collider: str = "hollow"    # "hollow" (compound, open interior) or "hull" (single solid convex hull)
    collider_wall: float = 2.5  # wall thickness of the hollow collider, mm (thicker than the real wall so
                                # small beads / particles can't tunnel; the interior is that much smaller)
    collider_sectors: int = 16  # angular pieces of the hollow collider
    spout: bool = False         # put a liquor pour spout (make_spout.py) in the neck
    spout_mass_g: float = 0.0   # >0: weighed spout mass; 0: computed from volumes (~27 g)

    @property
    def total_h(self) -> float:
        return self.body_h + self.shoulder_h + self.neck_h


# ----------------------------------------------------------------------------- profile


def _arc(cx, cz, r, a0, a1, n):
    return [(cx + r * math.cos(a), cz + r * math.sin(a)) for a in np.linspace(a0, a1, n)]


def _shoulder(p: BottleParams, inset: float, n: int = 24):
    """Shoulder curve from body radius down to neck radius, offset inwards by `inset`.

    Returns (points from body side to neck side, function r->z on the curve)."""
    R, rn = p.body_d / 2, p.neck_d / 2 - (p.thread_depth if p.threads else 0.0)
    z_a, z_b = p.body_h, p.body_h + p.shoulder_h
    if p.shoulder == "sphere":
        # sphere centred on the axis passing through (R, z_a) and (rn, z_b)
        z0 = ((rn**2 + z_b**2) - (R**2 + z_a**2)) / (2 * (z_b - z_a))
        Rs = math.hypot(R, z_a - z0) - inset
        a0 = math.atan2(z_a - z0, R)
        a1 = math.atan2(z_b - z0, rn)
        pts = _arc(0.0, z0, Rs, a0, a1, n)
        return pts, (lambda r: z0 + math.sqrt(max(Rs**2 - r**2, 0.0)))
    if p.shoulder == "ellipse":
        # quarter ellipse centred at (rn, z_a): tangent to the body, flat where it meets the neck
        a, b = R - rn - inset, p.shoulder_h - inset
        ts = np.linspace(0, math.pi / 2, n)
        pts = [(rn + a * math.cos(t), z_a + b * math.sin(t)) for t in ts]
        return pts, (lambda r: z_a + b * math.sqrt(max(1 - ((r - rn) / a) ** 2, 0.0)))
    raise ValueError(f"unknown shoulder style {p.shoulder!r}")


def outer_profile(p: BottleParams, with_threads: bool):
    """(r, z, sharp) points of the outside surface, bottom centre to top lip (mm)."""
    R, f = p.body_d / 2, p.base_fillet
    root = p.neck_d / 2 - (p.thread_depth if p.threads else 0.0)
    z_s, z_top = p.body_h + p.shoulder_h, p.total_h
    pts = [(0.0, 0.0, False), (R - f, 0.0, False)]
    pts += [(r, z, False) for r, z in _arc(R - f, f, f, -math.pi / 2, 0, 10)[1:]]
    pts += [(R, p.body_h, p.shoulder == "sphere")]  # crease where sphere meets cylinder
    sh, _ = _shoulder(p, 0.0)
    pts += [(r, z, False) for r, z in sh[1:-1]]
    pts += [(root, z_s, True)]
    if with_threads and p.threads:
        # trapezoidal ridges, kept clear of the shoulder and the lip
        band0, band1 = z_s + 3.0, z_top - 1.5
        pitch = (band1 - band0) / p.thread_count
        for i in range(p.thread_count):
            z0 = band0 + i * pitch
            w = pitch * 0.55
            ramp = w * 0.3
            crest = root + p.thread_depth
            pts += [(root, z0, True), (crest, z0 + ramp, True),
                    (crest, z0 + w - ramp, True), (root, z0 + w, True)]
    pts += [(root, z_top, True)]
    return pts


def full_profile(p: BottleParams):
    """Closed shell profile: outside up, across the lip, inside down to the bottom centre."""
    outer = outer_profile(p, with_threads=True)
    R, f, w = p.body_d / 2, p.base_fillet, p.wall
    root = p.neck_d / 2 - (p.thread_depth if p.threads else 0.0)
    rin_neck = root - p.neck_wall
    z_top = p.total_h

    inner_sh, z_of_r = _shoulder(p, w)
    # neck bore meets the inner shoulder surface where r = rin_neck
    z_bore = z_of_r(rin_neck)
    if z_bore >= z_top - 0.5:
        raise ValueError("neck wall too thick for this shoulder; reduce --neck-wall")
    sh_in = [(r, z) for r, z in inner_sh if r > rin_neck + 1e-6]
    z_body_in = sh_in[0][1]  # where inner cylinder meets inner shoulder

    inner = [(rin_neck, z_top, True), (rin_neck, z_bore, True)]
    inner += [(r, z, False) for r, z in reversed(sh_in[1:])]
    inner += [(R - w, z_body_in, p.shoulder == "sphere")]
    fi = max(f - w, 0.5)
    arc = _arc(R - w - fi, p.base_wall + fi, fi, 0, -math.pi / 2, 8)
    inner += [(r, z, False) for r, z in arc]
    inner += [(0.0, p.base_wall, False)]
    return outer + inner


# ----------------------------------------------------------------------------- revolve


def _seg_normals(prof):
    """Outward 2D normals of each profile segment (profile runs CCW around the material)."""
    out = []
    for (r0, z0, *_), (r1, z1, *_) in zip(prof[:-1], prof[1:]):
        dr, dz = r1 - r0, z1 - z0
        L = math.hypot(dr, dz) or 1.0
        out.append((dz / L, -dr / L))  # rotate tangent -90 deg
    return out


def revolve(prof, segments, cap_top=False):
    """Revolve an (r, z, sharp) profile around Z. Returns points, faceVertexCounts,
    faceVertexIndices and faceVarying normals (all in mm)."""
    prof = [(r, z, s) for r, z, s in prof]
    if cap_top:  # close a solid at the top with a pole on the axis
        prof = prof + [(0.0, prof[-1][1], True)]
    segn = _seg_normals(prof)
    thetas = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(thetas), np.sin(thetas)

    points, ring = [], []
    for r, z, _ in prof:
        if r < 1e-9:
            ring.append([len(points)] * segments)
            points.append((0.0, 0.0, z))
        else:
            ring.append(list(range(len(points), len(points) + segments)))
            points += [(r * c, r * s, z) for c, s in zip(cos, sin)]

    def pnorm(i, seg):  # 2D normal to use at profile point i on segment seg
        sharp = prof[i][2]
        if sharp or prof[i][0] < 1e-9:
            return segn[seg]
        nbrs = [segn[k] for k in (i - 1, i) if 0 <= k < len(segn)]
        nr, nz = sum(n[0] for n in nbrs), sum(n[1] for n in nbrs)
        L = math.hypot(nr, nz) or 1.0
        return (nr / L, nz / L)

    counts, idx, normals = [], [], []
    for k in range(len(prof) - 1):
        a, b = ring[k], ring[k + 1]
        na, nb = pnorm(k, k), pnorm(k + 1, k)
        for j in range(segments):
            j2 = (j + 1) % segments
            quad = [a[j], a[j2], b[j2], b[j]]
            angs = [j, j2, j2, j]
            ns = [na, na, nb, nb]
            # drop duplicate pole vertices -> triangles
            face, fn = [], []
            for v, ang, (nr, nz) in zip(quad, angs, ns):
                if face and face[-1] == v:
                    continue
                face.append(v)
                fn.append((nr * cos[ang], nr * sin[ang], nz))
            if len(face) > 1 and face[0] == face[-1]:
                face.pop(); fn.pop()
            if len(face) < 3:
                continue
            counts.append(len(face)); idx += face; normals += fn
    return np.array(points), counts, idx, normals


# ----------------------------------------------------------------------------- hollow collider


def hollow_collider(p: BottleParams, band_deg: float = 15.0):
    """Convex pieces (mm point arrays) forming a thick-walled open container, plus the
    interior profile [(r, z)] from the floor up to the rim."""
    t = p.collider_wall
    outer = [(r, z) for r, z, _ in outer_profile(p, with_threads=False)]
    # wall polyline starts where the outer profile reaches z = t (top of the floor piece)
    i = next(k for k, (_, z) in enumerate(outer) if z >= t)
    (r0, z0), (r1, z1) = outer[i - 1], outer[i]
    pt_t = (r0 + (r1 - r0) * (t - z0) / (z1 - z0), t) if z1 > t else outer[i]
    wall = [pt_t] + outer[i + (1 if z1 <= t else 0):]
    wall = [q for k, q in enumerate(wall) if k == 0 or math.dist(q, wall[k - 1]) > 1e-6]

    segs = []
    for (a, b) in zip(wall[:-1], wall[1:]):
        dr, dz = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dr, dz)
        segs.append((dz / L, -dr / L))  # outward normal (profile runs upward, material on the left)
    inner = []
    for k, (r, z) in enumerate(wall):
        ns = [segs[j] for j in (k - 1, k) if 0 <= j < len(segs)]
        nr, nz = sum(n[0] for n in ns), sum(n[1] for n in ns)
        L = math.hypot(nr, nz)
        nr, nz = nr / L, nz / L
        miter = min(1.0 / max(nr * ns[0][0] + nz * ns[0][1], 1e-3), 2.0)  # keep walls t thick at corners
        inner.append((r - nr * t * miter, z - nz * t * miter))

    # group consecutive segments into bands that turn less than band_deg
    bands, start = [], 0
    ang = [math.atan2(n[1], n[0]) for n in segs]
    for k in range(1, len(segs) + 1):
        if k == len(segs) or abs(math.degrees(ang[k] - ang[start])) > band_deg:
            bands.append((start, k))  # point indices start..k
            start = k
    n = p.collider_sectors
    th = np.linspace(0, 2 * math.pi, n + 1)
    pieces = []
    for a, b in bands:
        ring = wall[a:b + 1] + inner[a:b + 1]
        for j in range(n):
            angs = (th[j], 0.5 * (th[j] + th[j + 1]), th[j + 1])
            pieces.append(np.array([(r * math.cos(q), r * math.sin(q), z) for r, z in ring for q in angs]))
    # floor: everything of the outer profile below z = t, revolved, capped at z = t
    floor_prof = [q for q in outer if q[1] < t] + [pt_t]
    angs = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    floor = [(r * math.cos(q), r * math.sin(q), z) for r, z in floor_prof for q in angs]
    floor += [(0.0, 0.0, t)]
    pieces.insert(0, np.array(floor))
    interior = [(0.0, t)] + inner
    return pieces, interior


def _hull_mesh(points_mm):
    from scipy.spatial import ConvexHull
    pts = np.unique(np.round(points_mm, 6), axis=0)
    h = ConvexHull(pts)
    used = np.unique(h.simplices)
    remap = {v: k for k, v in enumerate(used)}
    tris = [[remap[v] for v in tri] for tri in h.simplices]
    return pts[used], [3] * len(tris), [v for tri in tris for v in tri]


def interior_volume_ml(interior) -> float:
    """Volume of revolution under the interior profile (r, z) from floor to rim, in ml."""
    rs, zs = np.array([q[0] for q in interior]), np.array([q[1] for q in interior])
    order = np.argsort(zs, kind="stable")
    rs, zs = rs[order], zs[order]
    return float(np.trapezoid(math.pi * rs**2, zs) / 1000) if hasattr(np, "trapezoid") else \
        float(np.trapz(math.pi * rs**2, zs) / 1000)


# ----------------------------------------------------------------------------- USD


def _set_mesh(mesh: UsdGeom.Mesh, pts_mm, counts, idx, normals=None):
    pts = pts_mm * MM
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    lo, hi = pts.min(0), pts.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    if normals is not None:
        mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)


def shell_mass_properties(pts_mm, counts, idx, mass_kg):
    """COM and diagonal inertia of the thin shell, scaled to `mass_kg`. Uses trimesh if present."""
    try:
        import trimesh
    except ImportError:
        return None
    tris = []
    for c, start in zip(counts, np.cumsum([0] + counts[:-1])):
        f = idx[start:start + c]
        tris += [[f[0], f[i], f[i + 1]] for i in range(1, c - 1)]
    m = trimesh.Trimesh(pts_mm * MM, np.array(tris), process=True)
    if not m.is_watertight:
        print("warning: render mesh not watertight, skipping inertia")
        return None
    if m.volume < 0:
        m.invert()
    m.density = mass_kg / m.volume
    I = m.moment_inertia  # about COM, principal axes align with XYZ by symmetry
    return m.center_mass, np.diag(I), m


def build(p: BottleParams, out_path: str):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetMetadata("comment", f"Procedural PET bottle: {vars(p)}")

    # --- materials
    looks = "/Bottle/Looks"
    UsdGeom.Scope.Define(stage, looks)
    glass = UsdShade.Material.Define(stage, f"{looks}/PET_Glass")
    mdl = UsdShade.Shader.Define(stage, f"{looks}/PET_Glass/OmniGlass")
    mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.98, 1.0))
    mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(p.ior)
    mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(p.frosting)
    mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)  # mesh has real wall thickness
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    glass.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    ps = UsdShade.Shader.Define(stage, f"{looks}/PET_Glass/PreviewSurface")  # usdview / other renderers
    ps.CreateIdAttr("UsdPreviewSurface")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.97, 1.0))
    ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.15)
    ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(p.ior)
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(max(p.frosting, 0.05))
    glass.CreateSurfaceOutput().ConnectToSource(ps.ConnectableAPI(), "surface")

    phys_mat = UsdShade.Material.Define(stage, f"{looks}/PET_Physics")
    pm = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    pm.CreateStaticFrictionAttr(p.static_friction)
    pm.CreateDynamicFrictionAttr(p.dynamic_friction)
    pm.CreateRestitutionAttr(p.restitution)

    # --- render mesh (hollow shell)
    vis_pts, vis_c, vis_i, vis_n = revolve(full_profile(p), p.segments)
    vis = UsdGeom.Mesh.Define(stage, "/Bottle/visuals")
    _set_mesh(vis, vis_pts, vis_c, vis_i, vis_n)
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(glass)

    # --- colliders (no threads; invisible)
    def add_collider(path, pts_mm, counts, idx):
        m = UsdGeom.Mesh.Define(stage, path)
        _set_mesh(m, pts_mm, counts, idx)
        m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        return len(counts)

    info = {}
    if p.collider == "hull":  # one solid convex hull of the outside
        col_pts, col_c, col_i, _ = revolve(outer_profile(p, with_threads=False), p.collision_segments,
                                           cap_top=True)
        col_faces = add_collider("/Bottle/collision", col_pts, col_c, col_i)
        info["collider"] = "solid convex hull"
    elif p.collider == "hollow":  # compound of convex wall pieces, open interior
        pieces, interior = hollow_collider(p)
        grp = UsdGeom.Xform.Define(stage, "/Bottle/collision")
        grp.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        col_faces = 0
        for k, pc in enumerate(pieces):
            col_faces += add_collider(f"/Bottle/collision/piece_{k:03d}", *_hull_mesh(pc))
        rm = Vt.FloatArray([q[0] * MM for q in interior])
        zm = Vt.FloatArray([q[1] * MM for q in interior])
        root.GetPrim().SetCustomDataByKey("interior_r_m", rm)
        root.GetPrim().SetCustomDataByKey("interior_z_m", zm)
        root.GetPrim().SetCustomDataByKey("collider_wall_m", float(p.collider_wall * MM))
        info["collider"] = f"hollow, {len(pieces)} convex pieces, wall {p.collider_wall} mm"
        info["collider_interior_ml"] = interior_volume_ml(interior)
    else:
        raise ValueError(f"unknown collider {p.collider!r}")

    # --- optional pour spout: part of the same rigid body
    spout_masses = []
    if p.spout:
        from make_spout import SpoutParams, add_spout
        spout_masses = add_spout(stage, "/Bottle/spout", SpoutParams(mass_g=p.spout_mass_g), p.total_h,
                                 neck_bore_d(p), phys_mat, looks)
        info["spout_g"] = sum(m for m, _ in spout_masses) * 1000

    # --- rigid body + mass
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_kg = p.mass_g / 1000
    props = shell_mass_properties(vis_pts, vis_c, vis_i, mass_kg)
    if props and spout_masses:  # combine: bottle shell + spout parts as point masses
        com_b, I_b, tm = props
        items = [(mass_kg, np.asarray(com_b), np.diag(I_b))]
        items += [(m, np.asarray(c) * MM, np.zeros((3, 3))) for m, c in spout_masses]
        M = sum(m for m, _, _ in items)
        com = sum(m * c for m, c, _ in items) / M
        I = sum(Ii + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com))
                for m, c, Ii in items)
        props = (com, np.diag(I), tm)  # off-diagonal terms from the bent nozzle are negligible
        mass_kg = M
    mass_api.CreateMassAttr(mass_kg)
    info["mass_kg"] = mass_kg
    if props:
        com, inertia, tm = props
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        info.update(com_m=com, inertia=inertia, watertight=tm.is_watertight,
                    shell_volume_ml=tm.volume * 1e6)
    stage.GetRootLayer().Save()
    info.update(render_faces=len(vis_c), collision_faces=col_faces,
                capacity_ml=capacity_ml(p))
    return info


def neck_bore_d(p: BottleParams) -> float:
    """Inside diameter of the neck finish, mm."""
    root = p.neck_d / 2 - (p.thread_depth if p.threads else 0.0)
    return 2 * (root - p.neck_wall)


def capacity_ml(p: BottleParams) -> float:
    """Inner volume via the shoelace/Pappus formula on the inner profile."""
    prof = full_profile(p)
    outer_n = len(outer_profile(p, with_threads=True))
    inner = [(r, z) for r, z, _ in prof[outer_n:]]
    inner = [(0.0, inner[0][1])] + inner  # close along the axis at the top
    # volume of revolution of a closed polygon = 2*pi * area * centroid_r
    A = Cr = 0.0
    for (r0, z0), (r1, z1) in zip(inner, inner[1:] + inner[:1]):
        cr = r0 * z1 - r1 * z0
        A += cr; Cr += (r0 + r1) * cr
    A /= 2
    return abs(2 * math.pi * (Cr / 6)) / 1000  # mm^3 -> ml


# ----------------------------------------------------------------------------- preview


def preview(p: BottleParams, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    prof = full_profile(p)
    rs, zs = [q[0] for q in prof], [q[1] for q in prof]
    ax.fill(rs, zs, color="#9cc9e6", ec="#1f5f8b", lw=0.8)
    ax.fill([-r for r in rs], zs, color="#9cc9e6", ec="#1f5f8b", lw=0.8)
    cp = outer_profile(p, with_threads=False)
    cr, cz = [q[0] for q in cp], [q[1] for q in cp]
    if p.collider == "hollow":
        _, interior = hollow_collider(p)
        ax.plot(cr, cz, "--", color="#d9534f", lw=1, label=f"collider (hollow, {p.collider_wall} mm wall)")
        ax.plot([q[0] for q in interior], [q[1] for q in interior], "--", color="#d9534f", lw=1)
    else:
        ax.plot(cr + [0], cz + [cz[-1]], "--", color="#d9534f", lw=1, label="collision hull")
    ax.axvline(0, color="#888", lw=0.5, ls=":")
    R = p.body_d / 2
    for z, lbl in [(p.body_h, f"body {p.body_h:g}"), (p.body_h + p.shoulder_h, f"+shoulder {p.shoulder_h:g}"),
                   (p.total_h, f"+neck {p.neck_h:g} = {p.total_h:g} mm")]:
        ax.axhline(z, color="#bbb", lw=0.5)
        ax.text(R + 2, z, lbl, va="center", fontsize=8)
    ax.set_title(f"Cross-section (Ø{p.body_d:g} body, Ø{p.neck_d:g} neck), mm\n"
                 f"wall {p.wall} / neck {p.neck_wall} / base {p.base_wall} — capacity ≈ {capacity_ml(p):.0f} ml",
                 fontsize=9)
    if p.spout:
        from make_spout import SpoutParams, draw_section
        draw_section(ax, SpoutParams(), neck_bore_d(p), z0=p.total_h)
    ax.set_aspect("equal"); ax.set_xlim(-R - 4, R + 26 + (15 if p.spout else 0))
    ax.legend(loc="lower right", fontsize=8)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    pts, counts, idx, _ = revolve(outer_profile(p, True), 64)
    polys, start = [], 0
    for c in counts:
        polys.append(pts[idx[start:start + c]]); start += c
    light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
    cols = []
    for poly in polys:
        n = np.cross(poly[1] - poly[0], poly[2] - poly[0]); L = np.linalg.norm(n) or 1
        s = 0.35 + 0.6 * max(np.dot(n / L, light), 0)
        cols.append((0.55 * s + 0.3, 0.75 * s + 0.2, 0.9 * s + 0.1, 0.55))
    ax3.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    h = p.total_h
    if p.spout:
        from make_spout import SpoutParams, parts
        base = {"rubber": np.array([0.12, 0.12, 0.13]), "steel": np.array([0.72, 0.74, 0.78])}
        for name, mat, sp, _, _, tris in parts(SpoutParams(), neck_bore_d(p)):
            if name in ("seal", "air_tube"):
                continue  # inside the bottle
            sp = sp + np.array([0.0, 0.0, p.total_h])
            poly = sp[tris]
            n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
            sh = (0.35 + 0.65 * np.clip(n @ light, 0, 1))[:, None]
            ax3.add_collection3d(Poly3DCollection(poly, facecolors=np.clip(base[mat] * (0.5 + sh), 0, 1),
                                                  edgecolors="none"))
        h = p.total_h + 60
    ax3.set_xlim(-h / 2, h / 2); ax3.set_ylim(-h / 2, h / 2); ax3.set_zlim(0, h)
    ax3.set_box_aspect((1, 1, 1)); ax3.view_init(elev=12, azim=-60); ax3.set_axis_off()
    ax3.set_title(f"Outer surface ({p.shoulder} shoulder, {'threads' if p.threads else 'no threads'})",
                  fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- CLI


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = BottleParams()
    for name, val in vars(d).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=val)
        elif name == "shoulder":
            ap.add_argument(flag, dest=name, choices=["sphere", "ellipse"], default=val)
        elif name == "collider":
            ap.add_argument(flag, dest=name, choices=["hollow", "hull"], default=val)
        else:
            ap.add_argument(flag, dest=name, type=type(val), default=val)
    ap.add_argument("-o", "--out", default="Bottle.usda")
    ap.add_argument("--preview", default=None, help="also write a PNG preview")
    a = vars(ap.parse_args())
    out, prev = a.pop("out"), a.pop("preview")
    p = BottleParams(**a)
    info = build(p, out)
    print(f"wrote {out}")
    for k, v in info.items():
        print(f"  {k}: {np.round(v, 7) if isinstance(v, np.ndarray) else v}")
    if prev:
        preview(p, prev)
        print(f"wrote {prev}")


if __name__ == "__main__":
    main()
