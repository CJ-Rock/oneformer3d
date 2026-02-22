#!/usr/bin/env python3
"""Convert synthetic rock-pile scene folders into .pth files.

Input scene folder format (from generate_synthetic_rockpile_lidar.py):
- scene_points.npy
- scene_instance_ids.npy
- lidar_points.npy
- lidar_view_ids.npy
- meta.json

Outputs:
- per-scene .pth files (default), or
- one aggregated .pth containing all scenes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except Exception:  # torch may not be installed in minimal envs
    torch = None


def save_pth(obj: Any, out_path: Path, prefer_torch: bool) -> str:
    """Save object into .pth using torch.save or pickle fallback."""
    if prefer_torch and torch is not None:
        torch.save(obj, out_path)
        return "torch"
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return "pickle"


def load_scene(scene_dir: Path, dtype: str = "float32") -> Dict[str, Any]:
    """Load one generated scene directory into a serializable dict."""
    points = np.load(scene_dir / "scene_points.npy").astype(dtype, copy=False)
    instance_ids = np.load(scene_dir / "scene_instance_ids.npy")
    lidar_points = np.load(scene_dir / "lidar_points.npy").astype(dtype, copy=False)
    lidar_view_ids = np.load(scene_dir / "lidar_view_ids.npy")
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))

    return {
        "scene_name": scene_dir.name,
        "points": points,
        "instance_ids": instance_ids,
        "lidar_points": lidar_points,
        "lidar_view_ids": lidar_view_ids,
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert synthetic rock-pile folders to .pth")
    parser.add_argument("--input-dir", type=str, required=True, help="Root folder with scene_xxxx directories")
    parser.add_argument("--output-dir", type=str, default="data/synth_rockpile_pth")
    parser.add_argument("--pattern", type=str, default="scene_*", help="Scene folder glob pattern")
    parser.add_argument("--aggregate", action="store_true", help="Store all scenes in one aggregated .pth")
    parser.add_argument("--aggregate-name", type=str, default="synth_rockpile_all.pth")
    parser.add_argument(
        "--save-backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "pickle"],
        help="auto: torch if available else pickle",
    )
    parser.add_argument(
        "--float-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Output dtype for point arrays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in in_dir.glob(args.pattern) if p.is_dir()])
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories matched: {in_dir / args.pattern}")

    if args.save_backend == "auto":
        prefer_torch = torch is not None
    elif args.save_backend == "torch":
        if torch is None:
            raise RuntimeError("--save-backend=torch requested, but torch is not installed.")
        prefer_torch = True
    else:
        prefer_torch = False

    if args.aggregate:
        all_scenes: List[Dict[str, Any]] = []
        for scene_dir in scene_dirs:
            all_scenes.append(load_scene(scene_dir, dtype=args.float_dtype))

        out_path = out_dir / args.aggregate_name
        backend = save_pth({"scenes": all_scenes}, out_path, prefer_torch=prefer_torch)
        print(f"saved aggregate: {out_path} ({backend}), scenes={len(all_scenes)}")
        return

    for scene_dir in scene_dirs:
        sample = load_scene(scene_dir, dtype=args.float_dtype)
        out_path = out_dir / f"{scene_dir.name}.pth"
        backend = save_pth(sample, out_path, prefer_torch=prefer_torch)
        print(
            f"saved: {out_path} ({backend}) "
            f"points={sample['points'].shape[0]} lidar={sample['lidar_points'].shape[0]}"
        )


if __name__ == "__main__":
    main()
