#!/usr/bin/env python3
"""Procedurally build a cut-crystal tumbler (Riedel DSG Neat style) as a sim-ready USD asset.

Shape (all in mm, Z up, origin at the centre of the glass's underside):
  - straight cylindrical outside, thick solid base, inner wall tapering from
    --wall-base at the floor to --wall-top at the rim, fully rounded rim
  - cut pattern on the outside: --flutes tall/short alternating V-cut spikes rising from an
    arched band above the base, and V-notches around the bottom edge (--no-cuts for a plain glass)

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
    flutes: int = 16             # number of spikes around (alternating tall / short)
    flute_depth: float = 2.4     # max depth of a spike cut at its base, mm
    tall_tip: float = 0.93       # tall spike tip height, fraction of height
    short_tip: float = 0.70      # short spike tip height, fraction of height
    flute_start: float = 24.0    # height where spikes start (arched band below), mm
    notch_depth: float = 2.5     # V-notch depth at the bottom edge, mm
    notch_h: float = 10.0        # V-notch height, mm
    density: float = 2.45        # g/cm^3 (lead-free crystal glass); mass = volume * density
    mass_g: float = 0.0          # >0 overrides the computed mass (weigh yours!)
    static_friction: float = 0.6
    dynamic_friction: float = 0.5
    restitution: float = 0.05
    ior: float = 1.52            # crystal glass
    segments_per_flute: int = 20  # angular resolution (render mesh = flutes * this)
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

    tag 'o' = outside (gets the cut pattern), 'n' = no displacement.
    """
    R, H, fo, fi, rr = p.R, p.height, p.bottom_fillet, p.floor_fillet, p.wall_top / 2
    pts = [(r, 0.0, "o") for r in np.linspace(0, R - fo, 6)]
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


def cut_depth(p: TumblerParams, theta: np.ndarray, z: float) -> np.ndarray:
    """Radial depth of the cut pattern (mm) at angles theta and height z, on the outside wall."""
    if not p.cuts:
        return np.zeros_like(theta)
    N, H = p.flutes, p.height
    pitch = 2 * math.pi / N
    d = np.zeros_like(theta)
    # spikes: V-grooves, widest at flute_start, tapering to a point at the tip
    for k in range(N):
        tall = k % 2 == 0
        tip = (p.tall_tip if tall else p.short_tip) * H
        w0 = (0.55 if tall else 0.45) * pitch
        z0 = p.flute_start
        if z >= tip:
            continue
        if z >= z0:
            t = (tip - z) / (tip - z0)
            w, depth = w0 * t, p.flute_depth * t ** 0.6  # tapers to a point, depth to 0
        else:  # rounded (arched) bottom end of the spike
            u = (z0 - z) / 5.0
            if u >= 1:
                continue
            w, depth = w0 * math.sqrt(1 - u * u), p.flute_depth * math.sqrt(1 - u * u)
        dth = np.angle(np.exp(1j * (theta - k * pitch)))
        d = np.maximum(d, depth * np.clip(1 - np.abs(dth) / max(w, 1e-6), 0, None))
    # V-notches around the bottom edge, between the spikes
    if z < p.notch_h:
        depth = p.notch_depth * (1 - z / p.notch_h)
        dth = np.angle(np.exp(1j * (theta - pitch / 2 - np.round((theta - pitch / 2) / pitch) * pitch)))
        d = np.maximum(d, depth * np.clip(1 - np.abs(dth) / (0.5 * pitch), 0, None))
    # never cut more than half the wall
    if z > p.base_h:
        d = np.minimum(d, 0.5 * (p.R - float(p.r_in(z))))
    if z > p.height - 4:
        d *= 0.0
    return d


# ----------------------------------------------------------------------------- mesh


def build_mesh(p: TumblerParams):
    prof = profile(p)
    S = p.flutes * p.segments_per_flute
    th = np.linspace(0, 2 * math.pi, S, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    points, rings = [], []
    for r, z, tag in prof:
        if r < 1e-9:
            rings.append([len(points)] * S)
            points.append((0.0, 0.0, z))
            continue
        rr = np.full(S, r)
        if tag == "o":
            rr = r - cut_depth(p, th, z) * (r / p.R)  # scales the underside too, so notches stay valid
        rings.append(list(range(len(points), len(points) + S)))
        points += list(zip(rr * c, rr * s, np.full(S, z)))
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

    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_subplot(1, 2, 1)
    prof = profile(p)
    rs = [q[0] for q in prof]
    zs = [q[1] for q in prof]
    ax.fill(rs, zs, color="#cfe3ef", ec="#1f5f8b", lw=0.8, label="section (between cuts)")
    th = np.array([0.0])  # through the centre of a tall spike / notch-free angle
    cut = [(q[0] - (cut_depth(p, th, q[1])[0] * q[0] / p.R if q[2] == "o" else 0), q[1]) for q in prof]
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
                 f"base {p.base_h:g} mm\nleft half through a spike cut; brim-full ≈ {capacity_ml(p):.0f} ml",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-p.R - 4, p.R + 26)
    ax.legend(loc="upper right", fontsize=7)

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    q = TumblerParams(**{**vars(p), "segments_per_flute": max(8, p.segments_per_flute // 2), "dz": 2.0})
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
    ax3.set_title(f"Outside ({q.flutes} spikes, alternating tall/short; notched base)"
                  if q.cuts else "Outside (no cuts)", fontsize=9)
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
