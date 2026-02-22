#!/usr/bin/env python3
"""Generate synthetic rock-pile point clouds and circular LiDAR scans.

This script creates multiple scenes made of irregular rock instances stacked into a pile,
then simulates 360° LiDAR scans from viewpoints distributed on an aerial circle.

Output format (per scene):
- scene_points.npy: full synthetic pile points (x, y, z)
- scene_instance_ids.npy: instance id per full-scene point
- lidar_points.npy: scanned points (x, y, z)
- lidar_view_ids.npy: scan view id per lidar point
- meta.json: generation parameters for reproducibility
- scene_instances.ply (optional): full scene point cloud, colorized by rock id
- lidar_scan.ply (optional): scanned points, colorized by view id
- scan_trajectory.xyz (optional): LiDAR origins on circular trajectory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def sample_irregular_rock(
    rng: np.random.Generator,
    n_surface_points: int,
    base_axes: np.ndarray,
    radial_noise: float,
) -> np.ndarray:
    """Sample an irregular ellipsoidal rock surface as a point cloud."""
    vec = rng.normal(size=(n_surface_points, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9

    noise = (
        1.0
        + radial_noise * np.sin(5.0 * vec[:, 0] + rng.uniform(-np.pi, np.pi))
        + radial_noise * 0.55 * np.sin(7.0 * vec[:, 1] + rng.uniform(-np.pi, np.pi))
        + radial_noise * 0.35 * np.sin(9.0 * vec[:, 2] + rng.uniform(-np.pi, np.pi))
    )
    pts = vec * noise[:, None] * base_axes[None, :]
    pts += rng.normal(scale=0.01 * np.mean(base_axes), size=pts.shape)
    return pts


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3x3 rotation matrix."""
    a, b, c = rng.uniform(0, 2 * np.pi, size=3)
    ca, cb, cc = np.cos([a, b, c])
    sa, sb, sc = np.sin([a, b, c])
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float32)
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)




def smooth_heightmap(hmap: np.ndarray, passes: int = 1) -> np.ndarray:
    """Lightweight 3x3 mean smoothing to reduce single-point spikes in support map."""
    out = hmap.copy()
    for _ in range(max(0, passes)):
        acc = out
        acc = acc + np.roll(out, 1, 0) + np.roll(out, -1, 0)
        acc = acc + np.roll(out, 1, 1) + np.roll(out, -1, 1)
        acc = acc + np.roll(np.roll(out, 1, 0), 1, 1)
        acc = acc + np.roll(np.roll(out, 1, 0), -1, 1)
        acc = acc + np.roll(np.roll(out, -1, 0), 1, 1)
        acc = acc + np.roll(np.roll(out, -1, 0), -1, 1)
        out = acc / 9.0
    return out

def compute_obb_dims(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute PCA-based OBB dimensions (x,y,z extents) and volume."""
    c = points.mean(axis=0, keepdims=True)
    x = points - c
    cov = np.cov(x.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    local = x @ evecs
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    dims = (maxs - mins).astype(np.float32)
    volume = float(np.prod(np.maximum(dims, 1e-6)))
    return dims, volume


def build_pile_scene(
    rng: np.random.Generator,
    n_rocks: int,
    points_per_rock_range: Tuple[int, int],
    pile_radius: float,
    grid_res: int,
    target_peak_height: float,
    drop_height: float,
    core_fill_ratio: float,
    core_radius_ratio: float,
    num_layers: int,
    collision_quantile: float,
    hmap_smooth_passes: int,
    upper_layer_spread: float,
    top_center_penalty: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Create one rock-pile scene and instance ids with natural mound-like stacking."""
    hmap = np.zeros((grid_res, grid_res), dtype=np.float32)
    xy_lim = pile_radius

    all_points = []
    all_instance_ids = []
    rock_infos = []

    def xy_to_idx(xy: np.ndarray) -> np.ndarray:
        uv = (xy + xy_lim) / (2 * xy_lim)
        uv = np.clip(uv, 0, 0.9999)
        return (uv * grid_res).astype(np.int32)

    # Bigger rocks first -> more realistic base.
    rough_sizes = rng.uniform(0.08, 0.22, size=(n_rocks, 3))
    size_order = np.argsort(np.prod(rough_sizes, axis=1))[::-1]

    # Layered stacking plan (e.g., 3층): base / middle / top.
    layer_edges = np.linspace(0, n_rocks, num_layers + 1, dtype=int)

    for seq, inst_id in enumerate(size_order.tolist()):
        n_pts = int(rng.integers(points_per_rock_range[0], points_per_rock_range[1] + 1))
        axes = rough_sizes[inst_id].astype(np.float32)
        axes[1] *= rng.uniform(0.8, 1.0)
        axes[2] *= rng.uniform(0.7, 0.95)

        rock = sample_irregular_rock(rng, n_pts, axes, radial_noise=rng.uniform(0.08, 0.22))
        rock = rock @ random_rotation(rng).T

        # Natural pile profile: high center, low edge.
        # Last small-rock phase explicitly targets inner-core filling.
        core_phase_start = int((1.0 - core_fill_ratio) * n_rocks)
        core_radius = pile_radius * core_radius_ratio

        layer_id = int(np.searchsorted(layer_edges[1:], seq, side="right"))
        layer_id = min(layer_id, num_layers - 1)
        # Core-fill on upper layers can over-concentrate points at center.
        # Restrict core-fill to lower/middle layers to keep realistic shoulder spread.
        in_core_fill_phase = (seq >= core_phase_start) and (layer_id < num_layers - 1)
        layer_frac = (layer_id + 1) / max(1, num_layers)
        layer_target = target_peak_height * layer_frac
        # Too small top-layer radius makes an unnatural central tower.
        # Keep upper layers sufficiently spread using configurable factor.
        layer_radius_scale = 1.0 - (1.0 - upper_layer_spread) * (layer_id / max(1, num_layers - 1))

        tries = 0
        best = None
        while tries < 35:
            tries += 1
            angle = rng.uniform(0, 2 * np.pi)
            if in_core_fill_phase:
                rad = abs(rng.normal(loc=0.0, scale=core_radius * 0.45))
                rad = min(rad, core_radius)
            else:
                rad = abs(rng.normal(loc=0.0, scale=(pile_radius * layer_radius_scale) * 0.42))
                rad = min(rad, (pile_radius * layer_radius_scale) * 0.98)
            tx, ty = rad * np.cos(angle), rad * np.sin(angle)

            rock_xy = rock[:, :2] + np.array([tx, ty], dtype=np.float32)
            idx = xy_to_idx(rock_xy)
            hmap_query = smooth_heightmap(hmap, passes=hmap_smooth_passes) if hmap_smooth_passes > 0 else hmap
            local_heights = hmap_query[idx[:, 1], idx[:, 0]]
            support_h = float(np.percentile(local_heights, 90))

            desired = layer_target * max(0.0, 1.0 - (rad / (pile_radius * layer_radius_scale + 1e-6)) ** 1.15)
            deficit = desired - support_h
            overshoot_penalty = max(0.0, support_h - desired) * 1.9
            center_bias = 0.12 * (rad / (pile_radius + 1e-6)) ** 2

            # If support is much lower than desired in the core, prefer filling it.
            core_void_bonus = 0.0
            if rad < core_radius:
                core_void_bonus = 0.35 * max(0.0, deficit)

            score = abs(deficit) + overshoot_penalty + center_bias - core_void_bonus
            if in_core_fill_phase:
                score += 0.04 * (rad / (core_radius + 1e-6))

            # Penalize too-central placement on top layer to avoid needle-like peak.
            if layer_id == num_layers - 1:
                top_core = pile_radius * core_radius_ratio
                if rad < top_core:
                    score += top_center_penalty * (1.0 - rad / (top_core + 1e-6))

            if best is None or score < best[0]:
                best = (score, tx, ty, support_h, rad, in_core_fill_phase)
            if score < 0.06:
                break

        _, tx, ty, support_h, rad, in_core_fill_phase = best
        # Gravity-drop placement: start 10cm above local support then fall until first collision.
        rock_xy = rock[:, :2] + np.array([tx, ty], dtype=np.float32)
        idx = xy_to_idx(rock_xy)
        hmap_query = smooth_heightmap(hmap, passes=hmap_smooth_passes) if hmap_smooth_passes > 0 else hmap
        local_surface = hmap_query[idx[:, 1], idx[:, 0]]
        rock_z = rock[:, 2]
        # Using strict max over sparse/noisy support can lift rocks unnaturally (apparent floating).
        # Use a high quantile to be penetration-safe but robust to single-cell spikes.
        q = float(np.clip(collision_quantile, 0.90, 1.0))
        tz_collision = float(np.quantile(local_surface - rock_z, q)) + 1e-3
        tz_start = float(support_h + drop_height - np.min(rock_z))
        tz = min(tz_start, tz_collision) if tz_start > tz_collision else tz_collision

        placed = rock + np.array([tx, ty, tz], dtype=np.float32)
        all_points.append(placed)
        all_instance_ids.append(np.full((placed.shape[0],), inst_id, dtype=np.int32))

        placed_idx = xy_to_idx(placed[:, :2])
        np.maximum.at(hmap, (placed_idx[:, 1], placed_idx[:, 0]), placed[:, 2])

        obb_dims, obb_volume = compute_obb_dims(placed)
        rock_infos.append(
            {
                "inst_id": int(inst_id),
                "placement_order": int(seq),
                "n_points": int(n_pts),
                "center_x": float(tx),
                "center_y": float(ty),
                "center_z": float(tz),
                "support_height": float(support_h),
                "drop_height": float(drop_height),
                "drop_start_z": float(tz_start),
                "fall_distance": float(max(0.0, tz_start - tz)),
                "collision_quantile": float(np.clip(collision_quantile, 0.90, 1.0)),
                "radial_distance": float(rad),
                "in_core_fill_phase": bool(in_core_fill_phase),
                "layer_id": int(layer_id),
                "num_layers": int(num_layers),
                "layer_radius_scale": float(layer_radius_scale),
                "top_center_penalty": float(top_center_penalty),
                "obb_x": float(obb_dims[0]),
                "obb_y": float(obb_dims[1]),
                "obb_z": float(obb_dims[2]),
                "obb_volume": float(obb_volume),
            }
        )

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    instance_ids = np.concatenate(all_instance_ids, axis=0)
    return points, instance_ids, rock_infos


def write_ascii_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Write xyzrgb point cloud in ASCII PLY format."""
    assert points.shape[0] == colors.shape[0], "points/colors size mismatch"
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def ids_to_colors(ids: np.ndarray, seed: int) -> np.ndarray:
    """Map integer ids to stable random RGB colors."""
    if ids.shape[0] == 0:
        return np.empty((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    n = int(ids.max()) + 1
    lut = rng.integers(30, 255, size=(n, 3), dtype=np.uint8)
    return lut[ids]




def circular_orbit_origins(
    num_views: int,
    scan_radius: float,
    scan_height: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Create uniformly spaced scanner origins on a circle at a fixed height."""
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False) + phase
    return np.stack(
        [
            scan_radius * np.cos(angles),
            scan_radius * np.sin(angles),
            np.full_like(angles, scan_height),
        ],
        axis=1,
    ).astype(np.float32)


def balance_points_per_view(
    lidar_points_list: List[np.ndarray],
    rng: np.random.Generator,
    max_points_per_view: int,
) -> List[np.ndarray]:
    """Keep per-view LiDAR points balanced by capping each view uniformly."""
    out=[]
    for pts in lidar_points_list:
        if max_points_per_view > 0 and pts.shape[0] > max_points_per_view:
            idx = rng.choice(pts.shape[0], size=max_points_per_view, replace=False)
            out.append(pts[idx])
        else:
            out.append(pts)
    return out

def lidar_scan_from_view(
    points: np.ndarray,
    origin: np.ndarray,
    n_azimuth: int,
    n_elevation: int,
    elev_min_deg: float,
    elev_max_deg: float,
    max_range: float,
) -> np.ndarray:
    """Simulate one 360° LiDAR scan using spherical binning with nearest hit."""
    rel = points - origin[None, :]
    d = np.linalg.norm(rel, axis=1)
    valid = (d > 1e-4) & (d < max_range)
    rel = rel[valid]
    d = d[valid]

    az = np.arctan2(rel[:, 1], rel[:, 0])
    el = np.degrees(np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1) + 1e-9))

    mask = (el >= elev_min_deg) & (el <= elev_max_deg)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    rel = rel[mask]
    d = d[mask]
    az = az[mask]
    el = el[mask]

    az_idx = np.floor((az + np.pi) / (2 * np.pi) * n_azimuth).astype(np.int32)
    az_idx = np.clip(az_idx, 0, n_azimuth - 1)
    el_idx = np.floor((el - elev_min_deg) / (elev_max_deg - elev_min_deg + 1e-9) * n_elevation).astype(np.int32)
    el_idx = np.clip(el_idx, 0, n_elevation - 1)

    lin = el_idx * n_azimuth + az_idx
    order = np.argsort(lin)
    lin_sorted = lin[order]
    d_sorted = d[order]

    _, first_idx = np.unique(lin_sorted, return_index=True)
    for i in range(len(first_idx)):
        s = first_idx[i]
        e = first_idx[i + 1] if i + 1 < len(first_idx) else len(lin_sorted)
        best_local = s + np.argmin(d_sorted[s:e])
        first_idx[i] = best_local

    chosen = order[first_idx]
    hits = rel[chosen] + origin[None, :]
    return hits.astype(np.float32)


def generate_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for scene_idx in range(args.num_scenes):
        scene_seed = int(rng.integers(0, 2**31 - 1))
        srng = np.random.default_rng(scene_seed)

        points, instance_ids, rock_infos = build_pile_scene(
            srng,
            n_rocks=args.num_rocks,
            points_per_rock_range=(args.min_pts_per_rock, args.max_pts_per_rock),
            pile_radius=args.pile_radius,
            grid_res=args.heightmap_res,
            target_peak_height=args.target_peak_height,
            drop_height=args.drop_height,
            core_fill_ratio=args.core_fill_ratio,
            core_radius_ratio=args.core_radius_ratio,
            num_layers=args.num_layers,
            collision_quantile=args.collision_quantile,
            hmap_smooth_passes=args.hmap_smooth_passes,
            upper_layer_spread=args.upper_layer_spread,
            top_center_penalty=args.top_center_penalty,
        )

        lidar_points = []
        lidar_view_ids = []
        scan_phase = float(srng.uniform(0, 2 * np.pi)) if args.random_scan_phase else 0.0
        origins = circular_orbit_origins(
            num_views=args.num_views,
            scan_radius=args.scan_radius,
            scan_height=args.scan_height,
            phase=scan_phase,
        )
        for view_id, origin in enumerate(origins):
            hits = lidar_scan_from_view(
                points,
                origin,
                n_azimuth=args.lidar_azimuth_bins,
                n_elevation=args.lidar_elevation_bins,
                elev_min_deg=args.lidar_elev_min,
                elev_max_deg=args.lidar_elev_max,
                max_range=args.lidar_max_range,
            )
            lidar_points.append(hits)

        lidar_points = balance_points_per_view(
            lidar_points, srng, max_points_per_view=args.max_points_per_view
        )
        for view_id, hits in enumerate(lidar_points):
            lidar_view_ids.append(np.full((hits.shape[0],), view_id, dtype=np.int32))

        lidar_points = np.concatenate(lidar_points, axis=0) if lidar_points else np.empty((0, 3), dtype=np.float32)
        lidar_view_ids = np.concatenate(lidar_view_ids, axis=0) if lidar_view_ids else np.empty((0,), dtype=np.int32)

        sdir = out_dir / f"scene_{scene_idx:04d}"
        sdir.mkdir(parents=True, exist_ok=True)

        np.save(sdir / "scene_points.npy", points)
        np.save(sdir / "scene_instance_ids.npy", instance_ids)
        np.save(sdir / "lidar_points.npy", lidar_points)
        np.save(sdir / "lidar_view_ids.npy", lidar_view_ids)

        if args.export_ply:
            scene_colors = ids_to_colors(instance_ids, seed=scene_seed)
            write_ascii_ply(points, scene_colors, sdir / "scene_instances.ply")
            if lidar_points.shape[0] > 0:
                lidar_colors = ids_to_colors(lidar_view_ids, seed=scene_seed + 17)
                write_ascii_ply(lidar_points, lidar_colors, sdir / "lidar_scan.ply")
            np.savetxt(sdir / "scan_trajectory.xyz", origins.astype(np.float32), fmt="%.6f")

        meta = {
            "scene_idx": scene_idx,
            "scene_seed": scene_seed,
            "num_rocks": args.num_rocks,
            "num_scene_points": int(points.shape[0]),
            "num_lidar_points": int(lidar_points.shape[0]),
            "num_views": args.num_views,
            "scan_radius": args.scan_radius,
            "scan_height": args.scan_height,
            "scan_phase": float(scan_phase),
            "max_points_per_view": int(args.max_points_per_view),
            "num_layers": args.num_layers,
            "per_view_counts": [int((lidar_view_ids == i).sum()) for i in range(args.num_views)],
            "rocks": rock_infos,
        }
        (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if not (args.min_total_points <= points.shape[0] <= args.max_total_points):
            print(
                f"[warn] scene_{scene_idx:04d} point count {points.shape[0]} out of "
                f"[{args.min_total_points}, {args.max_total_points}]"
            )

        if (scene_idx + 1) % max(1, args.log_every) == 0:
            print(
                f"[{scene_idx + 1}/{args.num_scenes}] "
                f"scene_pts={points.shape[0]} lidar_pts={lidar_points.shape[0]} rocks={args.num_rocks}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic rock-pile LiDAR dataset.")
    parser.add_argument("--output-dir", type=str, default="data/synth_rockpile")
    parser.add_argument("--num-scenes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-rocks", type=int, default=200, help="Rocks per pile.")
    parser.add_argument("--min-pts-per-rock", type=int, default=500)
    parser.add_argument("--max-pts-per-rock", type=int, default=1000)
    parser.add_argument("--min-total-points", type=int, default=100000)
    parser.add_argument("--max-total-points", type=int, default=200000)
    parser.add_argument("--pile-radius", type=float, default=1.5)
    parser.add_argument("--heightmap-res", type=int, default=220)
    parser.add_argument("--target-peak-height", type=float, default=1.2)
    parser.add_argument("--drop-height", type=float, default=0.1, help="Initial drop height above support (m).")
    parser.add_argument("--core-fill-ratio", type=float, default=0.35, help="Final fraction of rocks dedicated to center-core filling.")
    parser.add_argument("--core-radius-ratio", type=float, default=0.35, help="Core fill target radius as pile_radius ratio.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of vertical stacking layers (default: 3).")
    parser.add_argument("--collision-quantile", type=float, default=0.98, help="Robust collision quantile for settling (0.90~1.0).")
    parser.add_argument("--hmap-smooth-passes", type=int, default=1, help="Support-map smoothing passes to suppress spike artifacts.")
    parser.add_argument("--upper-layer-spread", type=float, default=0.80, help="Upper-layer radial spread ratio to avoid central towering.")
    parser.add_argument("--top-center-penalty", type=float, default=0.20, help="Penalty strength for top-layer over-central placement.")

    parser.add_argument("--num-views", type=int, default=24)
    parser.add_argument("--random-scan-phase", action="store_true", help="Randomize orbit start angle per scene while keeping uniform spacing.")
    parser.add_argument("--max-points-per-view", type=int, default=6000, help="Cap per-view hits for balanced coverage; <=0 disables.")
    parser.add_argument("--scan-radius", type=float, default=4.0)
    parser.add_argument("--scan-height", type=float, default=2.0)

    parser.add_argument("--lidar-azimuth-bins", type=int, default=1024)
    parser.add_argument("--lidar-elevation-bins", type=int, default=64)
    parser.add_argument("--lidar-elev-min", type=float, default=-35.0)
    parser.add_argument("--lidar-elev-max", type=float, default=10.0)
    parser.add_argument("--lidar-max-range", type=float, default=20.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--export-ply",
        action="store_true",
        help="Export visualization files (scene_instances.ply, lidar_scan.ply, scan_trajectory.xyz).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
