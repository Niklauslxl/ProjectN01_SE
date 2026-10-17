#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch 数据集与 DataLoader 封装

关键设计:
  1. 固定长度随机裁剪, 便于组 batch; 不足则补零, 评测时配合长度掩码使用
  2. **SNR 重混合(remix)**: VoiceBank 原始训练集只有 0/5/10/15dB 四档 SNR,
     很容易让模型记住噪声。训练时对每条数据重新混合随机 SNR, 泛化明显更好。
  3. 变长 batch 的 collate, 返回 lengths 供后续掩码

自检:
  python dataset.py                 # 路径与超参全部来自 config/*.yaml
  python dataset.py --split valid   # 检查验证集清单
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import add_config_args, load_config, pick, resolve_paths

EPS = 1e-8


def load_manifest(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["n_samples"] = int(r["n_samples"])
        r["duration"] = float(r["duration"])
    return rows


def _read(path: str) -> torch.Tensor:
    x, _ = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:  # 多声道取平均
        x = x.mean(axis=1)
    return torch.from_numpy(np.ascontiguousarray(x))


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2) + EPS)


def _crop_or_pad(x: torch.Tensor, target: int, start: int) -> torch.Tensor:
    """从 start 开始裁 target 个采样点, 不足则补零。clean / noisy 必须用同一个 start。"""
    seg = x[start:start + target]
    if seg.numel() < target:
        seg = F.pad(seg, (0, target - seg.numel()))
    return seg


def remix_snr(clean: torch.Tensor, noisy: torch.Tensor, snr_range=(0.0, 20.0)) -> torch.Tensor:
    """把 (clean, noisy) 的噪声分量重新加权到随机的目标 SNR。"""
    noise = noisy - clean
    target_snr = random.uniform(*snr_range)
    alpha = (torch.pow(torch.tensor(10.0), -target_snr / 20.0) * _rms(clean) / (_rms(noise) + EPS))
    return clean + alpha * noise


class VoiceBankDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        segment_sec: float = 3.0,
        sample_rate: int = 16000,
        training: bool = True,
        remix: bool = True,
        snr_range: tuple[float, float] = (0.0, 20.0),
        random_gain: tuple[float, float] = (0.7, 1.0),
        repeat_if_short: bool = True,
    ):
        self.rows = load_manifest(Path(csv_path))
        self.segment = int(segment_sec * sample_rate)
        self.training = training
        self.remix = remix and training
        self.snr_range = snr_range
        self.random_gain = random_gain
        self.repeat_if_short = repeat_if_short

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        clean = _read(row["clean_path"])
        noisy = _read(row["noisy_path"])

        n = min(clean.numel(), noisy.numel())
        clean, noisy = clean[:n], noisy[:n]

        # 短句重复拼接后再裁剪, 避免补零比例过高
        if self.repeat_if_short and self.training and n < self.segment:
            reps = (self.segment // n) + 1
            reps = min(reps, 4)
            clean = clean.repeat(reps)[: n * reps]
            noisy = noisy.repeat(reps)[: n * reps]

        if self.remix:
            noisy = remix_snr(clean, noisy, self.snr_range)

        if self.training:
            gain = random.uniform(*self.random_gain)
            clean, noisy = clean * gain, noisy * gain

        orig_len = min(clean.numel(), self.segment)
        start = random.randint(0, max(0, clean.numel() - self.segment)) if self.training else 0
        # clean 与 noisy 必须共用同一个裁剪起点, 否则标签会错位
        clean = _crop_or_pad(clean, self.segment, start)
        noisy = _crop_or_pad(noisy, self.segment, start)
        return {"clean": clean, "noisy": noisy, "length": orig_len, "utt_id": row["utt_id"]}


def collate(batch: list[dict]) -> dict:
    return {
        "clean": torch.stack([b["clean"] for b in batch]),
        "noisy": torch.stack([b["noisy"] for b in batch]),
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "utt_id": [b["utt_id"] for b in batch],
    }


def make_loader(csv_path: str | Path, batch_size: int = 16, num_workers: int = 4, **kwargs) -> DataLoader:
    dataset = VoiceBankDataset(csv_path, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.get("training", True),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=kwargs.get("training", True),
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="检查数据加载是否正常")
    add_config_args(ap)
    ap.add_argument("--data", type=Path, default=None, help="覆盖 config 里的 paths.data")
    ap.add_argument("--split", type=str, default="train", help="检查哪个清单(csv 文件名)")
    ap.add_argument("--segment", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)

    # --data 给了就用它, 否则用 config 的 paths.data; wav/manifest 目录名也来自 config
    data_root = Path(args.data) if args.data is not None else P.data
    manifest = data_root / P.manifest.relative_to(P.data) if data_root != P.data else P.manifest
    csv_path = manifest / f"{args.split}{cfg.paths.manifest_suffix}"
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}，请先运行 prepare.py")

    print(f"配置: data={data_root}  清单={csv_path}")
    loader = make_loader(
        csv_path,
        batch_size=pick(args.batch_size, cfg.dataset.batch_size),
        num_workers=pick(args.num_workers, cfg.dataset.num_workers),
        segment_sec=pick(args.segment, cfg.dataset.segment),
        sample_rate=pick(args.sr, cfg.data.sr),
        training=True,
        remix=cfg.dataset.remix,
        snr_range=tuple(cfg.dataset.snr_range),
        random_gain=tuple(cfg.dataset.random_gain),
        repeat_if_short=cfg.dataset.repeat_if_short,
    )
    batch = next(iter(loader))
    print(f"batch: clean={tuple(batch['clean'].shape)}  noisy={tuple(batch['noisy'].shape)}")
    print(f"lengths: {batch['length'].tolist()[:8]}")
    snr = 20 * torch.log10(
        torch.linalg.vector_norm(batch["clean"], dim=1)
        / (torch.linalg.vector_norm(batch["noisy"] - batch["clean"], dim=1) + EPS)
    )
    print(f"remix 后 SNR(dB): {[round(float(v), 1) for v in snr]}")


if __name__ == "__main__":
    main()
