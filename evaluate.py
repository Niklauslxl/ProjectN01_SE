#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试集评估: SI-SNR / PESQ(WB) / STOI, 可选导出增强后的 wav

  python evaluate.py --ckpt runs/vbd_base/best.pt --data ./data
  python evaluate.py --ckpt runs/vbd_base/best.pt --data ./data --dump ./enhanced --limit 100

注意: pesq / pystoi 为可选依赖; Windows 上 pip install pesq 需要 VS 生成工具,
装不上时会跳过对应指标, SI-SNR 始终可用。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch

from config import add_config_args, load_config, pick, resolve_paths
from dataset import _read, load_manifest
from train_demo import MaskUNet, si_snr as _si_snr


def si_snr_value(est: torch.Tensor, ref: torch.Tensor) -> float:
    return float(_si_snr(est, ref).mean())


def try_pesq(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float | None:
    """优先用官方 pesq(Cython 实现)，装不上时回退到纯 Python 的 pypesq。"""
    try:
        from pesq import pesq
        return float(pesq(sr, ref.numpy(), est.numpy(), "wb" if sr == 16000 else "nb"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from pypesq import pesq as pesq_py
        return float(pesq_py(ref.numpy(), est.numpy(), sr))
    except Exception:  # noqa: BLE001
        return None


def try_stoi(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float | None:
    try:
        from pystoi import stoi
        return float(stoi(ref.numpy(), est.numpy(), sr, extended=False))
    except Exception:  # noqa: BLE001
        return None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--ckpt", type=Path, default=None, help="权重路径, 默认 <paths.runs>/best.pt")
    ap.add_argument("--data", type=Path, default=None, help="覆盖 config 的 paths.data")
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--dump", type=Path, default=None, help="导出增强后音频的目录, 不填则不导出")
    ap.add_argument("--split", type=str, default=None, help="评估哪个子集, 默认 test")
    ap.add_argument("--limit", type=int, default=None, help="只评估前 N 条, 0 表示全部")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt is not None else Path(pick(cfg.eval.ckpt, P.runs / "best.pt"))
    if not ckpt_path.exists():
        raise SystemExit(f"找不到权重 {ckpt_path}（改 config 的 eval.ckpt 或传 --ckpt）")
    sr = int(pick(args.sr, cfg.eval.sr))
    limit = int(pick(args.limit, cfg.eval.limit))
    split = str(pick(args.split, cfg.paths.splits[2]))
    device_arg = str(pick(args.device, cfg.eval.device))
    dump_dir = Path(args.dump) if args.dump is not None else (
        Path(cfg.eval.dump) if cfg.eval.dump else None)

    device = torch.device(
        "cuda" if (device_arg == "auto" and torch.cuda.is_available())
        else (torch.device(device_arg) if device_arg != "auto" else "cpu")
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MaskUNet(
        n_fft=int(cfg.model.n_fft), hop=int(cfg.model.hop), win=int(cfg.model.win),
        channels=tuple(cfg.model.channels),
    ).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    data_root = Path(args.data) if args.data is not None else P.data
    manifest = data_root / P.manifest.relative_to(P.data) if data_root != P.data else P.manifest
    csv_path = manifest / f"{split}{cfg.paths.manifest_suffix}"
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}，请先运行 prepare.py")

    print(f"权重: {ckpt_path}\n清单: {csv_path}")
    rows = load_manifest(csv_path)
    if limit > 0:
        rows = rows[:limit]
    kind_clean, kind_noisy = (list(cfg.paths.kinds) + ["clean", "noisy"])[:2]
    if dump_dir:
        (dump_dir / kind_clean).mkdir(parents=True, exist_ok=True)
        (dump_dir / kind_noisy).mkdir(parents=True, exist_ok=True)
        (dump_dir / "est").mkdir(parents=True, exist_ok=True)
        print(f"导出目录: {dump_dir}")

    acc = {"base_si_snr": [], "si_snr": [], "pesq": [], "base_pesq": [], "stoi": [], "base_stoi": []}
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        clean = _read(row["clean_path"])
        noisy = _read(row["noisy_path"])
        n = min(clean.numel(), noisy.numel())
        clean, noisy = clean[:n], noisy[:n]

        est = model(noisy.unsqueeze(0).to(device)).squeeze(0).cpu()
        if est.numel() < n:
            est = torch.nn.functional.pad(est, (0, n - est.numel()))
        est = est[:n]

        acc["base_si_snr"].append(si_snr_value(noisy, clean))
        acc["si_snr"].append(si_snr_value(est, clean))
        if i <= 20 or len(rows) <= 20:  # PESQ 较慢, 只对前 20 条算
            p = try_pesq(clean, noisy, sr)
            pe = try_pesq(clean, est, sr)
            if p is not None and pe is not None:
                acc["base_pesq"].append(p)
                acc["pesq"].append(pe)
            s = try_stoi(clean, est, sr)
            sb = try_stoi(clean, noisy, sr)
            if s is not None and sb is not None:
                acc["base_stoi"].append(sb)
                acc["stoi"].append(s)

        if dump_dir:
            name = f"{row['utt_id']}.wav"
            sf.write(dump_dir / kind_clean / name, clean.numpy(), sr)
            sf.write(dump_dir / kind_noisy / name, noisy.numpy(), sr)
            sf.write(dump_dir / "est" / name, est.numpy(), sr)

        if i % 200 == 0:
            print(f"  已评估 {i}/{len(rows)}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 52)
    print(f"测试集 {len(rows)} 条, 用时 {time.time()-t0:.0f}s")
    print(f"SI-SNR : {mean(acc['base_si_snr']):.2f} dB -> {mean(acc['si_snr']):.2f} dB")
    if acc["pesq"]:
        print(f"PESQ   : {mean(acc['base_pesq']):.3f} -> {mean(acc['pesq']):.3f}  (前 {len(acc['pesq'])} 条)")
    else:
        print("PESQ   : 未安装 pesq 包, 已跳过 (pip install pesq)")
    if acc["stoi"]:
        print(f"STOI   : {mean(acc['base_stoi']):.3f} -> {mean(acc['stoi']):.3f}")
    else:
        print("STOI   : 未安装 pystoi 包, 已跳过 (pip install pystoi)")


if __name__ == "__main__":
    main()
