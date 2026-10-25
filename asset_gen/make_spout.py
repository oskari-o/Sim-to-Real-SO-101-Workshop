#!/usr/bin/env python3
"""Procedural liquor pour spout (stainless nozzle on a rubber stopper) for the PET bottle.

Used by make_bottle.py --spout, which puts it on the bottle's neck as part of the bottle's rigid
body. Can also be run on its own to write a standalone Spout.usda and a preview.

Spout frame (mm): z = 0 is the top of the bottle's neck lip, +z up along the bottle axis; the
nozzle bends towards +x.
  flange       black rubber disc sitting on the neck lip            (--flange-d x --flange-h)
  collar       stainless collar on the flange                       (--collar-d x --collar-h)
  nozzle       hollow tapered stainless tube, straight then bent, tip cut square (--nozzle-len, --bend-deg ...)
  seal         rubber stopper inside the neck, ribbed with conical fins, squeezed to the neck bore
  air tube     thin stainless tube running down into the bottle
All parts keep a through channel (--channel-d) so liquid can actually flow out of the nozzle.

  python3 make_spout.py -o Spout.usda --preview spout.png
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
DENSITY = {"rubber": 1.2, "steel": 7.9}  # g/cm^3


@dataclass
class SpoutParams:
    flange_d: float = 28.0       # rubber flange (sheet) on the neck lip
    flange_h: float = 3.0
    collar_d: float = 24.0       # stainless collar ("stem") on the flange
    collar_h: float = 4.0
    nozzle_len: float = 47.0     # along the centreline, from the collar top to the middle of the tip
    nozzle_d0: float = 8.0       # outer diameter at the collar
    nozzle_d1: float = 5.0       # outer diameter at the tip
    nozzle_wall: float = 0.6
    nozzle_straight: float = 10.0  # straight part before the bend
    bend_len: float = 14.0       # length of the bend
    bend_deg: float = 10.0       # bend angle, towards +x
    tip_cut_deg: float = 0.0     # tip cut slant from perpendicular (0 = square to the tube; >0: outer side of the bend longer)
    seal_len: float = 24.0       # rubber seal going down into the neck
    seal_core_d: float = 11.0
    seal_fin_d: float = 24.0     # free fin diameter; squeezed to the neck bore when inserted
    seal_fins: int = 4
    channel_d: float = 6.0       # liquid channel through flange, collar and seal
    air_tube_d: float = 3.0
    air_tube_len: float = 55.0   # measured down from the nozzle stem (top of the collar)
    air_tube_offset: float = 3.6  # from the axis, opposite the bend
    segments: int = 64           # circumferential resolution
    mass_g: float = 0.0          # >0 overrides the mass computed from volumes and densities


# ----------------------------------------------------------------------------- geometry


def _loop_mesh(loop, vertex_fn, segments):
    """Sweep a closed 2D loop [(rho, a)] around a circle: vertex_fn(rho, a, phi) -> xyz (mm).

    Returns points, faceVertexCounts, faceVertexIndices, triangles (closed, consistently oriented).
    """
    phis = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    K, S = len(loop), segments
    pts = np.array([vertex_fn(rho, a, phi) for rho, a in loop for phi in phis], float)
    quads = []
    for k in range(K):
        k2 = (k + 1) % K
        for j in range(S):
            j2 = (j + 1) % S
            quads.append([k * S + j, k * S + j2, k2 * S + j2, k2 * S + j])
    quads = np.array(quads)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    if np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() < 0:
        quads, tris = quads[:, ::-1], tris[:, ::-1]
    return pts, [4] * len(quads), quads.ravel().tolist(), tris


def _revolved(loop, segments, x0=0.0):
    """Loop of (r, z) revolved about a vertical axis through (x0, 0)."""
    return _loop_mesh(loop, lambda r, z, ph: (x0 + r * math.cos(ph), r * math.sin(ph), z), segments)


def _rounded_rect(r0, r1, z0, z1, rad, n=4):
    """Closed (r, z) loop of an annulus cross-section, outer corners rounded."""
    loop = [(r0, z0), (r1 - rad, z0)]
    loop += [(r1 - rad + rad * math.sin(a), z0 + rad - rad * math.cos(a)) for a in np.linspace(0, math.pi / 2, n)[1:]]
    loop += [(r1 - rad + rad * math.cos(a), z1 - rad + rad * math.sin(a)) for a in np.linspace(0, math.pi / 2, n)]
    loop += [(r0, z1)]
    return loop


def nozzle_path(p: SpoutParams, s: float):
    """Centreline point, tangent and in-plane normal (towards the bend) at arc length s (mm)."""
    z0 = p.flange_h + p.collar_h
    th = math.radians(p.bend_deg)
    s1, s2 = p.nozzle_straight, p.nozzle_straight + p.bend_len
    if s <= s1:
        c, a = np.array([0.0, 0.0, z0 + s]), 0.0
    elif s <= s2:
        R = p.bend_len / th
        a = (s - s1) / R
        c = np.array([R * (1 - math.cos(a)), 0.0, z0 + s1 + R * math.sin(a)])
    else:
        R = p.bend_len / th
        cb = np.array([R * (1 - math.cos(th)), 0.0, z0 + s1 + R * math.sin(th)])
        a = th
        c = cb + (s - s2) * np.array([math.sin(th), 0.0, math.cos(th)])
    t = np.array([math.sin(a), 0.0, math.cos(a)])
    n = np.array([math.cos(a), 0.0, -math.sin(a)])
    return c, t, n


def nozzle_mesh(p: SpoutParams):
    """Hollow tapered tube along the bent path; the tip is cut on a slanted plane."""
    L, tan_c = p.nozzle_len, math.tan(math.radians(p.tip_cut_deg))
    ro = lambda u: 0.5 * (p.nozzle_d0 + (p.nozzle_d1 - p.nozzle_d0) * u)  # noqa: E731
    us = np.linspace(0, 1, 48)
    # loop in (rho_sign, u): outer surface base->tip, tip face, inner surface tip->base, base face
    loop = [(+1, u) for u in us] + [(-1, u) for u in us[::-1]]

    def vert(side, u, ph):
        r_out = ro(u)
        r = r_out if side > 0 else r_out - p.nozzle_wall
        # each generator line has its own length so the tip lies on the cut plane
        s_end = L + tan_c * r * math.cos(ph)
        c, t, n = nozzle_path(p, u * s_end)
        return c + r * (math.cos(ph) * n + math.sin(ph) * np.array([0.0, 1.0, 0.0]))

    return _loop_mesh(loop, vert, p.segments)


def seal_loop(p: SpoutParams, bore_d: float):
    """(r, z) loop of the ribbed seal below the lip (z <= 0), with the liquid channel."""
    rc, rk, L = p.channel_d / 2, p.seal_core_d / 2, p.seal_len
    rf = min(p.seal_fin_d, bore_d) / 2 - 0.1  # fins squeezed into the bore
    loop = [(rc, 0.0), (rk, 0.0)]
    pitch = (L - 3.0) / p.seal_fins
    for k in range(p.seal_fins):
        zt = -1.5 - k * pitch
        loop += [(rk, zt), (rf, zt - 0.6), (rf, zt - 1.6), (rk, zt - min(3.2, pitch - 0.3))]
    loop += [(rk, -L + 1.2), (rk - 1.2, -L), (rc, -L)]
    return loop[::-1]  # any orientation works; _loop_mesh fixes it


def air_tube_bottom(p: SpoutParams) -> float:
    """z of the air tube's lower end: --air-tube-len below the nozzle stem (never above the seal)."""
    return min(p.flange_h + p.collar_h - p.air_tube_len, -p.seal_len - 1.0)


def parts(p: SpoutParams, bore_d: float):
    """[(name, material, points_mm, counts, idx, tris)] in the spout frame."""
    rc = p.channel_d / 2
    out = []
    out.append(("flange", "rubber", *_revolved(_rounded_rect(rc, p.flange_d / 2, 0.0, p.flange_h, 0.8), p.segments)))
    out.append(("collar", "steel", *_revolved(
        _rounded_rect(rc, p.collar_d / 2, p.flange_h, p.flange_h + p.collar_h, 0.6), p.segments)))
    out.append(("nozzle", "steel", *nozzle_mesh(p)))
    out.append(("seal", "rubber", *_revolved(seal_loop(p, bore_d), p.segments)))
    ra = p.air_tube_d / 2
    zb = air_tube_bottom(p)
    tube = [(ra * 0.4, -p.seal_len + 2), (ra, -p.seal_len + 2), (ra, zb), (ra * 0.4, zb)]
    out.append(("air_tube", "steel", *_revolved(tube, 16, x0=-p.air_tube_offset)))
    return out


def volume_com(pts, tris):
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6
    return vol.sum(), (vol[:, None] * (v0 + v1 + v2) / 4).sum(0) / vol.sum()


def collider_pieces(p: SpoutParams, bore_d: float):
    """Convex pieces (mm, spout frame) leaving the liquid channel and nozzle bore open."""
    rc, pieces = p.channel_d / 2, []

    def ring_sectors(r0, r1, z0, z1, n):
        th = np.linspace(0, 2 * math.pi, n + 1)
        for j in range(n):
            a = (th[j], 0.5 * (th[j] + th[j + 1]), th[j + 1])
            pieces.append(np.array([(r * math.cos(q), r * math.sin(q), z) for r in (r0, r1)
                                    for q in a for z in (z0, z1)]))

    ring_sectors(rc, p.flange_d / 2, 0.0, p.flange_h + p.collar_h, 10)
    ring_sectors(rc, min(p.seal_fin_d, bore_d) / 2 - 0.1, -p.seal_len, 0.0, 8)
    nseg, nsec = 5, 6
    ss = np.linspace(0, p.nozzle_len, nseg + 1)
    th = np.linspace(0, 2 * math.pi, nsec + 1)
    for a, b in zip(ss[:-1], ss[1:]):
        for j in range(nsec):
            pts = []
            for s in (a, b):
                c, t, n = nozzle_path(p, s)
                u = s / p.nozzle_len
                r_out = 0.5 * (p.nozzle_d0 + (p.nozzle_d1 - p.nozzle_d0) * u)
                for r in (r_out, r_out - p.nozzle_wall):
                    for q in (th[j], 0.5 * (th[j] + th[j + 1]), th[j + 1]):
                        pts.append(c + r * (math.cos(q) * n + math.sin(q) * np.array([0, 1.0, 0])))
            pieces.append(np.array(pts))
    return pieces


# ----------------------------------------------------------------------------- USD


def _materials(stage, looks):
    mats = {}
    for key, color, metallic, rough in (("rubber", (0.02, 0.02, 0.022), 0.0, 0.75),
                                        ("steel", (0.78, 0.78, 0.8), 1.0, 0.22)):
        m = UsdShade.Material.Define(stage, f"{looks}/Spout_{key}")
        sh = UsdShade.Shader.Define(stage, f"{looks}/Spout_{key}/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        mats[key] = m
    return mats


def add_spout(stage, path: str, p: SpoutParams, z_mm: float, bore_d: float, phys_mat=None, looks=None):
    """Add the spout under `path`, translated to z_mm in the parent frame.

    Colliders get CollisionAPI only (no rigid body): under a rigid-body prim they become part of
    it. Returns [(mass_kg, com_mm_in_parent)] per part for the caller's mass properties.
    """
    looks = looks or f"{path}/Looks"
    UsdGeom.Scope.Define(stage, looks)
    mats = _materials(stage, looks)
    grp = UsdGeom.Xform.Define(stage, path)
    grp.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_mm * MM))
    masses, offset = [], np.array([0.0, 0.0, z_mm])
    total_free = 0.0
    for name, mat, pts, counts, idx, tris in parts(p, bore_d):
        m = UsdGeom.Mesh.Define(stage, f"{path}/{name}")
        _set_mesh(m, pts, counts, idx)
        fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
        vn = np.zeros_like(pts)
        for i in range(3):
            np.add.at(vn, tris[:, i], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats[mat])
        vol, com = volume_com(pts, tris)
        g = vol / 1000 * DENSITY[mat]  # mm^3 -> cm^3 -> g
        total_free += g
        masses.append([g, com + offset])
    if p.mass_g > 0:  # rescale to the weighed mass, keeping the distribution
        for mm in masses:
            mm[0] *= p.mass_g / total_free
    cg = UsdGeom.Xform.Define(stage, f"{path}/collision")
    cg.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    for k, pc in enumerate(collider_pieces(p, bore_d)):
        cp, cc, ci = _hull_mesh(pc)
        cm = UsdGeom.Mesh.Define(stage, f"{path}/collision/piece_{k:03d}")
        _set_mesh(cm, cp, cc, ci)
        cm.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cm.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(cm.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        if phys_mat is not None:
            UsdShade.MaterialBindingAPI.Apply(cm.GetPrim()).Bind(
                phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return [(g / 1000, c) for g, c in masses]


def draw_section(ax, p: SpoutParams, bore_d: float, z0: float = 0.0):
    """Draw the spout's cross-section (xz plane) on a matplotlib axis, shifted up by z0 mm."""
    for loop, col in ((_rounded_rect(p.channel_d / 2, p.flange_d / 2, 0, p.flange_h, 0.8), "#222"),
                      (_rounded_rect(p.channel_d / 2, p.collar_d / 2, p.flange_h, p.flange_h + p.collar_h, 0.6), "#999"),
                      (seal_loop(p, bore_d), "#222")):
        r = np.array([q[0] for q in loop])
        z = np.array([q[1] for q in loop]) + z0
        for sgn in (1, -1):
            ax.fill(sgn * r, z, color=col, alpha=0.85)
    ss = np.linspace(0, p.nozzle_len, 60)
    for side in (1, -1):
        for wall in (0, p.nozzle_wall):
            xs, zs = [], []
            for s in ss:
                c, t, n = nozzle_path(p, s)
                r = 0.5 * (p.nozzle_d0 + (p.nozzle_d1 - p.nozzle_d0) * s / p.nozzle_len) - wall
                q = c + side * r * n
                xs.append(q[0])
                zs.append(q[2] + z0)
            ax.plot(xs, zs, color="#666", lw=1)
    ax.plot([-p.air_tube_offset] * 2, [z0 - p.seal_len, z0 + air_tube_bottom(p)], color="#999", lw=2)


def preview(p: SpoutParams, path: str, bore_d: float = 16.0, neck_d: float = 20.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    base = {"rubber": np.array([0.12, 0.12, 0.13]), "steel": np.array([0.72, 0.74, 0.78])}
    for name, mat, pts, counts, idx, tris in parts(p, bore_d):
        poly = pts[tris]
        n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
        sh = (0.35 + 0.65 * np.clip(n @ light, 0, 1))[:, None]
        cols = np.clip(base[mat] * (0.5 + sh) + (0.25 * sh ** 8 if mat == "steel" else 0), 0, 1)
        ax.add_collection3d(Poly3DCollection(poly, facecolors=cols, edgecolors="none"))
    # neck of the bottle, for scale
    th = np.linspace(0, 2 * math.pi, 40)
    for z in np.linspace(-16, 0, 5):
        ax.plot(neck_d / 2 * np.cos(th), neck_d / 2 * np.sin(th), z, color="#7fb3d5", lw=0.6)
    ax.set_xlim(-30, 40)
    ax.set_ylim(-35, 35)
    ax.set_zlim(air_tube_bottom(p) - 5, 60)
    ax.set_box_aspect((70, 70, 65 - air_tube_bottom(p)))
    ax.view_init(elev=8, azim=-90)
    ax.set_axis_off()
    ax.set_title("Pour spout (blue: bottle neck, for scale)", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    draw_section(ax2, p, bore_d)
    rn, rb = neck_d / 2, bore_d / 2
    ax2.fill([rb, rn, rn, rb], [-16, -16, 0, 0], color="#7fb3d5", alpha=0.5)
    ax2.fill([-rb, -rn, -rn, -rb], [-16, -16, 0, 0], color="#7fb3d5", alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"Section: flange Ø{p.flange_d:g}×{p.flange_h:g}, collar Ø{p.collar_d:g}×{p.collar_h:g}, "
                  f"nozzle {p.nozzle_len:g} mm / {p.bend_deg:g}°, seal {p.seal_len:g} mm in a Ø{bore_d:g} bore",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, val in vars(SpoutParams()).items():
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=type(val), default=val)
    ap.add_argument("--bore-d", type=float, default=16.0, help="neck bore the seal sits in, mm")
    ap.add_argument("-o", "--out", default="Spout.usda")
    ap.add_argument("--preview", default=None)
    a = vars(ap.parse_args())
    out, prev, bore = a.pop("out"), a.pop("preview"), a.pop("bore_d")
    p = SpoutParams(**a)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Spout")
    stage.SetDefaultPrim(root.GetPrim())
    masses = add_spout(stage, "/Spout/geom", p, 0.0, bore)
    stage.GetRootLayer().Save()
    print(f"wrote {out}; mass {sum(m for m, _ in masses) * 1000:.1f} g "
          f"({', '.join(f'{n} {m * 1000:.1f}' for (n, *_), (m, _) in zip(parts(p, bore), masses))})")
    if prev:
        preview(p, prev, bore)
        print(f"wrote {prev}")


if __name__ == "__main__":
    main()
