#!/usr/bin/env python3
"""Procedurally build a cut-crystal tumbler (Riedel DSG Neat style) as a sim-ready USD asset.

Shape (all in mm, Z up, origin at the centre of the glass's underside):
  - straight cylindrical outside, thick solid base, inner wall tapering from
    --wall-base at the floor to --wall-top at the rim, fully rounded rim
  - cut pattern on the outside (--no-cuts for a plain glass): --dents V-cut dents, each starting
    as a --notch-width wide triangular notch in the bottom edge, narrowing up to --hull-h (sides
    nearly straight at first, then curving in, (1 - u^a)^b), then running straight up as a narrow
    --groove-width groove that fades out just below the rim
  - underside: concave (cosine bell, --punt-depth higher at the centre, resting on a ring at the
    edge), with a radial V-cut from each side dent's notch to a smooth --centre-d disc

USD layout (same conventions as make_bottle.py):
  /Tumbler             Xform, defaultPrim, RigidBodyAPI + MassAPI (mass, COM, inertia from the mesh)
  /Tumbler/visuals     closed render mesh with real wall thickness, crystal glass material
  /Tumbler/collision   invisible compound collider: solid base + convex wall sectors, open interior
  customData interior_r_m / interior_z_m: inside profile (floor to rim), used by utils/bottle_fill.py

  python3 make_tumbler.py -o Tumbler.usda --preview tumbler.png
  python3 make_tumbler.py --height 77 --diameter 70 --no-cuts
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_bottle import _hull_mesh, _set_mesh  # noqa: E402

MM = 1e-3


@dataclass
class TumblerParams:
    height: float = 82.0         # overall height, mm
    diameter: float = 82.0       # outer diameter, mm
    base_h: float = 13.0         # solid base: inside floor height, mm
    wall_base: float = 6.0       # wall thickness just above the floor, mm
    wall_top: float = 2.0        # wall thickness at the rim, mm (rim is a full round)
    floor_fillet: float = 4.0    # inside floor-to-wall fillet radius, mm
    bottom_fillet: float = 1.0   # outside bottom edge fillet, mm
    cuts: bool = True            # model the cut-crystal pattern
    dents: int = 15              # number of V-cut dents around
    notch_depth: float = 2.5     # dent depth at the bottom edge, mm
    notch_width: float = 11.0    # dent width at the bottom edge, mm
    hull_h: float = 28.0         # height over which the dent narrows from the notch to the groove, mm
    hull_a: float = 2.4          # taper shape (1 - u^a)^b, u = z / hull_h: larger a keeps the sides
    hull_b: float = 1.2          #   straighter for longer before curving in; b > 1 eases into the groove
    groove_width: float = 2.0    # width of the straight groove above the hull part, mm
    groove_depth: float = 0.8    # depth of the straight groove, mm
    groove_fade: float = 10.0    # length over which the groove narrows and fades out at the top, mm
    top_margin: float = 3.0      # groove ends this far below the top of the rim, mm
    punt_depth: float = 3.0      # underside is concave: centre this much higher than the resting ring, mm
    centre_d: float = 10.0       # smooth disc in the middle of the underside where the radial cuts end, mm
    bottom_cuts: bool = True     # radial V-cuts on the underside, one per side dent
    density: float = 2.45        # g/cm^3 (lead-free crystal glass); mass = volume * density
    mass_g: float = 0.0          # >0 overrides the computed mass (weigh yours!)
    static_friction: float = 0.6
    dynamic_friction: float = 0.5
    restitution: float = 0.05
    ior: float = 1.52            # crystal glass
    res_mm: float = 1.0          # angular resolution along the circumference, mm (4x finer around each groove)
    dz: float = 1.0              # vertical resolution of the outer wall, mm
    collider_sectors: int = 16

    @property
    def R(self) -> float:
        return self.diameter / 2

    def r_in(self, z):
        """Inside radius of the (straight, tapered) inner wall at height z."""
        z0, z1 = self.base_h, self.height - self.wall_top / 2
        t = np.clip((np.asarray(z, float) - z0) / (z1 - z0), 0, 1)
        return self.R - self.wall_base + (self.wall_base - self.wall_top) * t


# ----------------------------------------------------------------------------- profile


def profile(p: TumblerParams):
    """Closed (r, z, tag) profile: underside -> outside wall -> rim -> inside wall -> floor.

    tag 'u' = underside (concave, radial cuts), 'o' = outside wall (side cuts), 'n' = untouched.
    """
    R, H, fo, fi, rr = p.R, p.height, p.bottom_fillet, p.floor_fillet, p.wall_top / 2
    n_u = max(8, int(round((R - fo) / p.res_mm)))
    pts = [(r, punt_z(p, r), "u") for r in np.linspace(0, R - fo, n_u + 1)]
    pts += [(R - fo + fo * math.sin(a), fo - fo * math.cos(a), "o") for a in np.linspace(0, math.pi / 2, 6)[1:]]
    top = H - rr - 1.0
    n = max(2, int(round((top - fo) / p.dz)))
    pts += [(R, z, "o") for z in np.linspace(fo, top, n + 1)[1:]]
    # rim: semicircle from the outside over to the inside
    cr, cz = R - rr, H - rr
    pts += [(cr + rr * math.cos(a), cz + rr * math.sin(a), "n") for a in np.linspace(0, math.pi, 12)]
    # inner wall down to the fillet
    zf = p.base_h + fi
    pts += [(float(p.r_in(z)), z, "n") for z in np.linspace(cz, zf, 8)[1:]]
    # inner fillet (approximately tangent to the wall) and flat floor
    rc = float(p.r_in(zf)) - fi
    pts += [(rc + fi * math.cos(a), zf - fi * math.sin(a), "n") for a in np.linspace(0, math.pi / 2, 7)[1:]]
    pts += [(r, p.base_h, "n") for r in np.linspace(rc, 0, 5)[1:]]
    return pts


def punt_z(p: TumblerParams, r: float) -> float:
    """Height of the concave underside at radius r: cosine bell, punt_depth at the centre, 0 at the edge."""
    edge = p.R - p.bottom_fillet
    return p.punt_depth * 0.5 * (1 + math.cos(math.pi * min(r, edge) / edge))


def bottom_cut(p: TumblerParams, theta: np.ndarray, r: float) -> np.ndarray:
    """Upward depth (mm) of the radial V-cuts on the underside at angles theta and radius r.

    Each cut lines up with a side dent: as wide as the notch at the rim, narrowing linearly (with
    its depth) to a point at the edge of the smooth centre disc. It eases out over the last few mm
    before the rim, where the side notch takes over.
    """
    rc = 0.5 * p.centre_d
    if not (p.cuts and p.bottom_cuts) or r <= rc:
        return np.zeros_like(theta)
    edge = p.R - p.bottom_fillet
    pitch = 2 * math.pi / p.dents
    t = (r - rc) / (p.R - rc)
    hw = 0.5 * min(p.notch_width, pitch * p.R) * t
    depth = p.notch_depth * t * float(np.clip((edge - r) / 3.0, 0, 1))
    ds = np.abs(theta - np.round(theta / pitch) * pitch) * r
    return depth * np.clip(1 - ds / max(hw, 1e-6), 0, None)


def underside(p: TumblerParams, theta: np.ndarray, r: float, z: float):
    """Displaced underside point(s): the side notch pulls the outer few mm inward, the radial cut lifts it."""
    edge_w = 5.0
    w = float(np.clip((r - (p.R - p.bottom_fillet - edge_w)) / edge_w, 0, 1))
    w = w * w * (3 - 2 * w)  # smoothstep
    rr = r - cut_depth(p, theta, 0.0) * w
    zz = z + bottom_cut(p, theta, r)
    return rr, zz


def dent_shape(p: TumblerParams, z: float) -> tuple[float, float]:
    """(half-width in mm along the circumference, depth in mm) of one dent at height z."""
    pitch_mm = 2 * math.pi * p.R / p.dents
    hw0, hg = 0.5 * min(p.notch_width, pitch_mm), 0.5 * p.groove_width
    z_end = p.height - p.top_margin
    if z >= z_end:
        return 0.0, 0.0
    if z < p.hull_h:  # notch -> groove, curved taper
        f = (1 - (z / p.hull_h) ** p.hull_a) ** p.hull_b  # gentle straight-ish taper, then curves in
        hw, depth = hg + (hw0 - hg) * f, p.groove_depth + (p.notch_depth - p.groove_depth) * f
    else:
        hw, depth = hg, p.groove_depth
    fade = min(1.0, (z_end - z) / max(p.groove_fade, 1e-6))  # narrows to a point and vanishes
    return hw * fade, depth * fade


def cut_depth(p: TumblerParams, theta: np.ndarray, z: float) -> np.ndarray:
    """Radial depth of the cut pattern (mm) at angles theta and height z, on the outside wall."""
    if not p.cuts:
        return np.zeros_like(theta)
    pitch = 2 * math.pi / p.dents
    hw, depth = dent_shape(p, z)
    if hw <= 0 or depth <= 0:
        return np.zeros_like(theta)
    # arc distance (mm) from the nearest dent centre
    ds = np.abs(theta - np.round(theta / pitch) * pitch) * p.R
    d = depth * np.clip(1 - ds / hw, 0, None)
    if z > p.base_h:  # never cut more than half the wall
        d = np.minimum(d, 0.5 * (p.R - float(p.r_in(z))))
    return d


def angles(p: TumblerParams) -> np.ndarray:
    """Circumferential samples: --res-mm spacing, 4x finer within the groove, apex/edges included."""
    pitch = 2 * math.pi / p.dents
    pitch_mm = pitch * p.R
    fine = max(p.res_mm / 4, 0.05)
    near = 0.5 * p.groove_width * 1.5
    s = np.concatenate([np.arange(-near, near + 1e-9, fine),
                        np.arange(near, pitch_mm / 2, p.res_mm),
                        -np.arange(near, pitch_mm / 2, p.res_mm), [pitch_mm / 2]])
    s = np.unique(np.round(s, 6))
    s = s[(s > -pitch_mm / 2 + 1e-6) & (s <= pitch_mm / 2 + 1e-9)]
    th = (np.arange(p.dents)[:, None] * pitch + (s / p.R)[None, :]).ravel()
    return np.sort(np.mod(th, 2 * math.pi))


# ----------------------------------------------------------------------------- mesh


def build_mesh(p: TumblerParams):
    prof = profile(p)
    th = angles(p)
    S = len(th)
    c, s = np.cos(th), np.sin(th)
    points, rings = [], []
    for r, z, tag in prof:
        if r < 1e-9:
            rings.append([len(points)] * S)
            points.append((0.0, 0.0, z))
            continue
        rr, zz = np.full(S, r), np.full(S, z)
        if tag == "o":
            rr = r - cut_depth(p, th, z) * (r / p.R)
        elif tag == "u":
            rr, zz = underside(p, th, r, z)
        rings.append(list(range(len(points), len(points) + S)))
        points += list(zip(rr * c, rr * s, zz))
    counts, idx = [], []
    for k in range(len(prof) - 1):
        a, b = rings[k], rings[k + 1]
        for j in range(S):
            j2 = (j + 1) % S
            face = []
            for v in (a[j], a[j2], b[j2], b[j]):
                if not face or face[-1] != v:
                    face.append(v)
            if len(face) > 1 and face[0] == face[-1]:
                face.pop()
            if len(face) >= 3:
                counts.append(len(face))
                idx += face
    pts = np.array(points)
    tris = []
    start = 0
    for n in counts:
        f = idx[start:start + n]
        tris += [[f[0], f[i], f[i + 1]] for i in range(1, n - 1)]
        start += n
    tris = np.array(tris)
    # orientation: make the enclosed volume positive (outward normals)
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6
    if vol < 0:
        tris = tris[:, ::-1]
        idx = [v for n, st in zip(counts, np.cumsum([0] + counts[:-1])) for v in idx[st:st + n][::-1]]
    # smooth vertex normals (area weighted)
    fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
    vn = np.zeros_like(pts)
    for i in range(3):
        np.add.at(vn, tris[:, i], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
    return pts, counts, idx, tris, vn


def mass_properties(pts_mm, tris, mass_g, density):
    try:
        import trimesh
    except ImportError:
        return None
    m = trimesh.Trimesh(pts_mm * MM, tris, process=True)
    if not m.is_watertight:
        print("warning: render mesh not watertight, skipping inertia")
        return None
    vol = m.volume
    mass = mass_g / 1000 if mass_g > 0 else vol * density * 1000  # g/cm^3 == 1000 kg/m^3
    m.density = mass / vol
    return mass, m.center_mass, np.diag(m.moment_inertia), vol


def collider_pieces(p: TumblerParams):
    """Solid base disc + convex wall sectors (from floor to rim), interior left open."""
    R, H, bh = p.R, p.height, p.base_h
    ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    base = [(R * math.cos(a), R * math.sin(a), z) for a in ang for z in (0.0, bh)]
    pieces = [np.array(base)]
    n = p.collider_sectors
    th = np.linspace(0, 2 * math.pi, n + 1)
    for j in range(n):
        angs = (th[j], 0.5 * (th[j] + th[j + 1]), th[j + 1])
        ring = [(R, bh), (R, H), (float(p.r_in(H)), H), (float(p.r_in(bh)), bh)]
        pieces.append(np.array([(r * math.cos(a), r * math.sin(a), z) for r, z in ring for a in angs]))
    interior = [(0.0, bh), (float(p.r_in(bh)), bh), (float(p.r_in(H)), H)]
    return pieces, interior


# ----------------------------------------------------------------------------- USD


def build(p: TumblerParams, out_path: str):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Tumbler")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetMetadata("comment", f"Procedural cut-crystal tumbler: {vars(p)}")

    looks = "/Tumbler/Looks"
    UsdGeom.Scope.Define(stage, looks)
    glass = UsdShade.Material.Define(stage, f"{looks}/Crystal")
    mdl = UsdShade.Shader.Define(stage, f"{looks}/Crystal/OmniGlass")
    mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.98, 0.99, 1.0))
    mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(p.ior)
    mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    glass.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    ps = UsdShade.Shader.Define(stage, f"{looks}/Crystal/PreviewSurface")
    ps.CreateIdAttr("UsdPreviewSurface")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.97, 1.0))
    ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.2)
    ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(p.ior)
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
    glass.CreateSurfaceOutput().ConnectToSource(ps.ConnectableAPI(), "surface")

    phys_mat = UsdShade.Material.Define(stage, f"{looks}/Glass_Physics")
    pm = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    pm.CreateStaticFrictionAttr(p.static_friction)
    pm.CreateDynamicFrictionAttr(p.dynamic_friction)
    pm.CreateRestitutionAttr(p.restitution)

    pts, counts, idx, tris, vn = build_mesh(p)
    vis = UsdGeom.Mesh.Define(stage, "/Tumbler/visuals")
    _set_mesh(vis, pts, counts, idx)
    vis.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))
    vis.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(glass)

    pieces, interior = collider_pieces(p)
    grp = UsdGeom.Xform.Define(stage, "/Tumbler/collision")
    grp.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    col_faces = 0
    for k, pc in enumerate(pieces):
        cp, cc, ci = _hull_mesh(pc)
        m = UsdGeom.Mesh.Define(stage, f"/Tumbler/collision/piece_{k:03d}")
        _set_mesh(m, cp, cc, ci)
        m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        col_faces += len(cc)
    root.GetPrim().SetCustomDataByKey("interior_r_m", Vt.FloatArray([q[0] * MM for q in interior]))
    root.GetPrim().SetCustomDataByKey("interior_z_m", Vt.FloatArray([q[1] * MM for q in interior]))

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    info = {}
    props = mass_properties(pts, tris, p.mass_g, p.density)
    if props:
        mass, com, inertia, vol = props
        mass_api.CreateMassAttr(float(mass))
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        info.update(mass_g=mass * 1000, glass_volume_ml=vol * 1e6, com_mm=com * 1000, inertia=inertia)
    else:
        mass_api.CreateMassAttr(float(p.mass_g / 1000 if p.mass_g > 0 else 0.3))
    stage.GetRootLayer().Save()
    info.update(capacity_ml=capacity_ml(p), render_faces=len(counts), collider_pieces=len(pieces),
                collision_faces=col_faces)
    return info


def capacity_ml(p: TumblerParams) -> float:
    """Brim-full capacity: tapered cavity from the floor to the top of the rim."""
    h = p.height - p.base_h
    r0, r1 = float(p.r_in(p.base_h)), float(p.r_in(p.height))
    return math.pi * h / 3 * (r0 * r0 + r0 * r1 + r1 * r1) / 1000


# ----------------------------------------------------------------------------- preview


def preview(p: TumblerParams, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(21, 6.2))
    ax = fig.add_subplot(1, 4, 1)
    prof = profile(p)
    rs = [q[0] for q in prof]
    zs = [q[1] for q in prof]
    ax.fill(rs, zs, color="#cfe3ef", ec="#1f5f8b", lw=0.8, label="section (between cuts)")
    th = np.array([0.0])  # through the centre of a dent
    cut = []
    for r, z, tag in prof:
        if tag == "o":
            cut.append((r - cut_depth(p, th, z)[0] * r / p.R, z))
        elif tag == "u":
            ru, zu = underside(p, th, r, z)
            cut.append((float(ru[0]), float(zu[0])))
        else:
            cut.append((r, z))
    ax.plot([-c[0] for c in cut], [c[1] for c in cut], color="#1f5f8b", lw=0.8)
    ax.fill([-c[0] for c in cut], [c[1] for c in cut], color="#cfe3ef")
    pieces, interior = collider_pieces(p)
    ax.plot([q[0] for q in interior], [q[1] for q in interior], "--", color="#d9534f", lw=1,
            label="collider interior")
    ax.axvline(0, color="#999", lw=0.5, ls=":")
    for z, lbl in [(p.base_h, f"floor {p.base_h:g}"), (p.height, f"rim {p.height:g} mm")]:
        ax.axhline(z, color="#bbb", lw=0.5)
        ax.text(p.R + 2, z, lbl, va="center", fontsize=8)
    ax.set_title(f"Section: Ø{p.diameter:g} × {p.height:g} mm, wall {p.wall_base:g}→{p.wall_top:g} mm, "
                 f"base {p.base_h:g} mm\nleft half through a dent; brim-full ≈ {capacity_ml(p):.0f} ml",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-p.R - 4, p.R + 26)
    ax.legend(loc="upper right", fontsize=7)

    # unrolled outside wall: cut depth over 3 dent spacings (outline of each dent)
    axm = fig.add_subplot(1, 4, 2)
    pitch_mm = 2 * math.pi * p.R / p.dents
    smm = np.linspace(-1.5 * pitch_mm, 1.5 * pitch_mm, 600)
    zz = np.linspace(0, p.height, 330)
    depth = np.array([cut_depth(p, smm / p.R, z) for z in zz])
    im = axm.pcolormesh(smm, zz, depth, cmap="Blues", shading="auto")
    axm.contour(smm, zz, depth, levels=[0.05], colors="#1f5f8b", linewidths=0.8)
    axm.axhline(p.hull_h, color="#d9534f", lw=0.6, ls="--")
    axm.text(smm[-1], p.hull_h + 1, f"hull_h {p.hull_h:g}", ha="right", fontsize=7, color="#d9534f")
    axm.set_aspect("equal")
    axm.set_xlabel("around the circumference, mm")
    axm.set_ylabel("height, mm")
    axm.set_title("Outside unrolled: cut depth (3 dents)", fontsize=9)
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.02, label="depth, mm")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    q = TumblerParams(**{**vars(p), "res_mm": max(p.res_mm, 1.5), "dz": 1.5})
    pts, counts, idx, tris, vn = build_mesh(q)
    outer = pts[tris].mean(axis=1)
    keep = np.hypot(outer[:, 0], outer[:, 1]) > q.R - 3.2  # outside surface only, for a clean picture
    tri = tris[keep]
    polys = pts[tri]
    n = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    light = np.array([0.5, -0.7, 0.5])
    light /= np.linalg.norm(light)
    sh = 0.3 + 0.7 * np.clip(n @ light, 0, 1)
    cols = np.stack([0.55 * sh + 0.35, 0.7 * sh + 0.25, 0.85 * sh + 0.15, np.full_like(sh, 0.9)], 1)
    ax3.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    h = q.height
    ax3.set_xlim(-h / 2, h / 2)
    ax3.set_ylim(-h / 2, h / 2)
    ax3.set_zlim(0, h)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=14, azim=-90)
    ax3.set_axis_off()
    ax3.set_title(f"Outside ({q.dents} dents: notch -> hull taper to {q.hull_h:g} mm -> "
                  f"{q.groove_width:g} mm groove)"
                  if q.cuts else "Outside (no cuts)", fontsize=9)

    # underside, seen from below
    axb = fig.add_subplot(1, 4, 4, projection="3d")
    cen = pts[tris].mean(axis=1)
    tri = tris[cen[:, 2] < q.punt_depth + q.notch_depth + 0.5]
    tri = tri[np.hypot(*pts[tri].mean(axis=1)[:, :2].T) < q.R - 0.3]
    polys = pts[tri]
    n = np.cross(polys[:, 1] - polys[:, 0], polys[:, 2] - polys[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    light = np.array([0.45, -0.35, -0.82])
    light /= np.linalg.norm(light)
    sh = 0.2 + 0.8 * np.clip(n @ light, 0, 1)
    cols = np.stack([0.55 * sh + 0.35, 0.7 * sh + 0.25, 0.85 * sh + 0.15, np.full_like(sh, 1.0)], 1)
    axb.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"))
    axb.set_xlim(-q.R, q.R)
    axb.set_ylim(-q.R, q.R)
    axb.set_zlim(-q.R + 6, q.R + 6)
    axb.set_box_aspect((1, 1, 1))
    axb.view_init(elev=-80, azim=-90)
    axb.set_axis_off()
    axb.set_title(f"Underside: {q.punt_depth:g} mm concave, {q.dents} radial cuts, "
                  f"Ø{q.centre_d:g} mm smooth centre", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(TumblerParams()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, dest=name, action=argparse.BooleanOptionalAction, default=val)
        else:
            ap.add_argument(flag, dest=name, type=type(val), default=val)
    ap.add_argument("-o", "--out", default="Tumbler.usda")
    ap.add_argument("--preview", default=None, help="also write a PNG preview")
    a = vars(ap.parse_args())
    out, prev = a.pop("out"), a.pop("preview")
    p = TumblerParams(**a)
    info = build(p, out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    for k, v in info.items():
        print(f"  {k}: {np.round(v, 7) if isinstance(v, np.ndarray) else (round(v, 2) if isinstance(v, float) else v)}")
    if prev:
        preview(p, prev)
        print(f"wrote {prev}")


if __name__ == "__main__":
    main()
