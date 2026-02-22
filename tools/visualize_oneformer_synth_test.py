#!/usr/bin/env python3
"""Visualize test results for OneFormer-style synthetic rockpile model.

Inputs:
- Test split .pth files under <data_root>/test
- Trained checkpoint from tools/train_oneformer_synth.py (best.pth/last.pth)

Outputs per scene:
- <scene>_gt.ply      : points colorized by GT label
- <scene>_pred.ply    : points colorized by predicted label
- <scene>_error.ply   : green(correct)/red(wrong)
- <scene>_meta.json   : per-scene metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def load_pth(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


class OneFormerPointSeg(nn.Module):
    def __init__(self, num_classes: int, d_model: int, nhead: int, num_layers: int, num_queries: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(points)
        key_padding = ~mask
        mem = self.encoder(x, src_key_padding_mask=key_padding)

        b = points.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        dec = self.decoder(tgt=q, memory=mem, memory_key_padding_mask=key_padding)

        attn = torch.einsum("bnd,bqd->bnq", mem, dec) / (mem.shape[-1] ** 0.5)
        q_cls = self.class_head(dec)
        logits = torch.einsum("bnq,bqc->bnc", attn.softmax(dim=-1), q_cls)
        return logits


def color_lut(num_classes: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 255, size=(num_classes, 3), dtype=np.uint8)


def write_ascii_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
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
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def infer_one(model: nn.Module, points: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        pts = torch.from_numpy(points.astype(np.float32))[None].to(device)
        mask = torch.ones((1, pts.shape[1]), dtype=torch.bool, device=device)
        logits = model(pts, mask)
        pred = logits.argmax(dim=-1)[0].cpu().numpy().astype(np.int64)
    return pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize OneFormer synthetic test predictions")
    p.add_argument("--data-root", type=str, required=True, help="Root containing test/ folder")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (best.pth/last.pth)")
    p.add_argument("--output-dir", type=str, default="work_dirs/synth_oneformer_vis")
    p.add_argument("--max-scenes", type=int, default=50, help="Visualize first N test scenes")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_pth(args.ckpt)
    cargs: Dict = ckpt.get("args", {})

    model = OneFormerPointSeg(
        num_classes=int(cargs.get("num_classes", 200)),
        d_model=int(cargs.get("d_model", 128)),
        nhead=int(cargs.get("nhead", 8)),
        num_layers=int(cargs.get("num_layers", 4)),
        num_queries=int(cargs.get("num_queries", 128)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(args.device)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.data_root, "test", "*.pth")))
    if not files:
        raise FileNotFoundError(f"No test .pth files in {os.path.join(args.data_root, 'test')}")
    files = files[: args.max_scenes]

    lut = color_lut(int(cargs.get("num_classes", 200)), seed=0)
    scene_metrics: List[Dict] = []

    for fp in files:
        sample = load_pth(fp)
        points = np.asarray(sample["points"], dtype=np.float32)
        gt = np.asarray(sample["instance_ids"], dtype=np.int64)
        num_classes = lut.shape[0]
        gt_mod = gt % num_classes

        pred = infer_one(model, points, args.device)
        ok = pred == gt_mod
        acc = float(ok.mean()) if ok.size else 0.0

        name = Path(fp).stem
        gt_col = lut[gt_mod]
        pred_col = lut[pred]
        err_col = np.zeros((points.shape[0], 3), dtype=np.uint8)
        err_col[ok] = np.array([0, 220, 0], dtype=np.uint8)
        err_col[~ok] = np.array([220, 0, 0], dtype=np.uint8)

        write_ascii_ply(points, gt_col, out_dir / f"{name}_gt.ply")
        write_ascii_ply(points, pred_col, out_dir / f"{name}_pred.ply")
        write_ascii_ply(points, err_col, out_dir / f"{name}_error.ply")

        meta = {"scene": name, "num_points": int(points.shape[0]), "acc": acc}
        (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        scene_metrics.append(meta)
        print(f"{name}: acc={acc:.4f} points={points.shape[0]}")

    summary = {
        "num_scenes": len(scene_metrics),
        "mean_acc": float(np.mean([m["acc"] for m in scene_metrics])) if scene_metrics else 0.0,
        "scenes": scene_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved visualization to {out_dir}")


if __name__ == "__main__":
    main()
