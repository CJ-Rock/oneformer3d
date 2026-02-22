#!/usr/bin/env python3
"""Train/validate a OneFormer-style point-seg model on converted synth .pth splits.

Expected folder layout (from convert_synth_rockpile_to_pth.py):
  <data_root>/train/*.pth
  <data_root>/val/*.pth

Each .pth should include at least:
  - points: (N,3)
  - instance_ids: (N,)
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_pth(path: str):
    """Robust .pth loader across torch versions and save backends.

    - PyTorch >=2.6 changed torch.load default to weights_only=True.
    - Our dataset .pth files may contain numpy objects, requiring weights_only=False.
    - If file is pure pickle (converter fallback), use pickle.load.
    """
    torch_load_sig = inspect.signature(torch.load)
    has_weights_only = "weights_only" in torch_load_sig.parameters

    # 1) Prefer torch.load with weights_only=False when available.
    try:
        if has_weights_only:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")
    except Exception:
        pass

    # 2) Try legacy torch.load call once more (for edge compatibility).
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        pass

    # 3) Fallback to raw pickle only for converter pickle backend outputs.
    with open(path, "rb") as f:
        return pickle.load(f)


class SynthRockpilePthDataset(Dataset):
    def __init__(self, split_dir: str, max_points: int, num_classes: int | None = None):
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.pth")))
        if not self.files:
            raise FileNotFoundError(f"No .pth files found under {split_dir}")
        self.max_points = max_points
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = load_pth(self.files[idx])
        points = np.asarray(d["points"], dtype=np.float32)
        labels = np.asarray(d["instance_ids"], dtype=np.int64)

        # Optional class cap for stable training when instance IDs are large.
        if self.num_classes is not None:
            labels = labels % self.num_classes

        if points.shape[0] > self.max_points:
            sel = np.random.choice(points.shape[0], self.max_points, replace=False)
            points = points[sel]
            labels = labels[sel]

        return {
            "points": torch.from_numpy(points),
            "labels": torch.from_numpy(labels),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    bsz = len(batch)
    nmax = max(x["points"].shape[0] for x in batch)

    points = torch.zeros((bsz, nmax, 3), dtype=torch.float32)
    labels = torch.full((bsz, nmax), -100, dtype=torch.long)
    mask = torch.zeros((bsz, nmax), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["points"].shape[0]
        points[i, :n] = item["points"]
        labels[i, :n] = item["labels"]
        mask[i, :n] = True

    return {"points": points, "labels": labels, "mask": mask}


class OneFormerPointSeg(nn.Module):
    """Lightweight OneFormer-style query decoder for point-wise segmentation."""

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
        # points: [B, N, 3], mask: [B, N] True=valid
        x = self.input_proj(points)
        key_padding = ~mask
        mem = self.encoder(x, src_key_padding_mask=key_padding)

        b = points.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        dec = self.decoder(tgt=q, memory=mem, memory_key_padding_mask=key_padding)

        # Point-wise logits via attention-like projection from points to decoded queries.
        attn = torch.einsum("bnd,bqd->bnq", mem, dec) / (mem.shape[-1] ** 0.5)
        q_cls = self.class_head(dec)  # [B,Q,C]
        logits = torch.einsum("bnq,bqc->bnc", attn.softmax(dim=-1), q_cls)
        return logits


def ap50_proxy(pred: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute a class-wise IoU>=0.5 hit ratio as AP50-like proxy for segmentation."""
    valid = labels != ignore_index
    if not valid.any():
        return 0.0
    p = pred[valid]
    g = labels[valid]
    classes = torch.unique(torch.cat([p, g], dim=0))
    hits = []
    for c in classes:
        tp = ((p == c) & (g == c)).sum().item()
        fp = ((p == c) & (g != c)).sum().item()
        fn = ((p != c) & (g == c)).sum().item()
        den = tp + fp + fn
        iou = (tp / den) if den > 0 else 0.0
        hits.append(1.0 if iou >= 0.5 else 0.0)
    return float(np.mean(hits)) if hits else 0.0


@dataclass
class Metrics:
    loss: float
    acc: float
    ap50: float


def step(model, batch, criterion, optimizer=None):
    points = batch["points"].cuda(non_blocking=True)
    labels = batch["labels"].cuda(non_blocking=True)
    mask = batch["mask"].cuda(non_blocking=True)

    logits = model(points, mask)
    loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        valid = labels != -100
        correct = (pred[valid] == labels[valid]).float().mean().item() if valid.any() else 0.0
        ap50 = ap50_proxy(pred, labels, ignore_index=-100)

    return Metrics(loss=float(loss.item()), acc=correct, ap50=ap50)


def run_epoch(model, loader, criterion, optimizer=None, desc: str = ""):
    train = optimizer is not None
    model.train(train)

    losses, accs, ap50s = [], [], []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(train):
        for batch in iterator:
            m = step(model, batch, criterion, optimizer)
            losses.append(m.loss)
            accs.append(m.acc)
            ap50s.append(m.ap50)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{np.mean(accs):.4f}", ap50=f"{np.mean(ap50s):.4f}")

    return Metrics(loss=float(np.mean(losses)), acc=float(np.mean(accs)), ap50=float(np.mean(ap50s)))


def main():
    parser = argparse.ArgumentParser(description="Train OneFormer-style model on synth rockpile .pth splits")
    parser.add_argument("--data-root", type=str, required=True, help="Root containing train/ and val/ folders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--num-classes", type=int, default=200)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=128)

    parser.add_argument("--save-dir", type=str, default="work_dirs/synth_oneformer")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    train_ds = SynthRockpilePthDataset(
        split_dir=os.path.join(args.data_root, "train"),
        max_points=args.max_points,
        num_classes=args.num_classes,
    )
    val_ds = SynthRockpilePthDataset(
        split_dir=os.path.join(args.data_root, "val"),
        max_points=args.max_points,
        num_classes=args.num_classes,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    model = OneFormerPointSeg(
        num_classes=args.num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_queries=args.num_queries,
    ).cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, desc=f"train {epoch:03d}")
        va = run_epoch(model, val_loader, criterion, optimizer=None, desc=f"val {epoch:03d}")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={tr.loss:.4f} train_acc={tr.acc:.4f} train_ap50={tr.ap50:.4f} "
            f"val_loss={va.loss:.4f} val_acc={va.acc:.4f} val_ap50={va.ap50:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_loss": va.loss,
            "val_ap50": va.ap50,
        }
        torch.save(ckpt, save_dir / "last.pth")

        if va.loss < best_val:
            best_val = va.loss
            torch.save(ckpt, save_dir / "best.pth")


if __name__ == "__main__":
    main()
