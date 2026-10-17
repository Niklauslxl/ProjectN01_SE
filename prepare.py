#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VoiceBank-DEMAND 数据准备脚本: 解压 -> 重采样 -> 划分验证集 -> 生成训练清单

产出目录结构:
    data/
      train/clean/*.wav   train/noisy/*.wav
      valid/clean/*.wav   valid/noisy/*.wav
      test/clean/*.wav    test/noisy/*.wav
      manifest/train.csv  valid.csv  test.csv  summary.json

clean 与 noisy 的 wav 文件名一一对应 (例如 p226_001.wav)。

用法:
  python prepare.py --downloads ./downloads --data ./data --sr 16000
  python prepare.py --downloads ./downloads --data ./data48k --sr 48000 --workers 16
  python prepare.py --downloads ./downloads --data ./data --val-ratio 0.1   # 随机划分验证集
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from config import add_config_args, load_config, pick, resolve_paths

# ---------------------------------------------------------------- 重采样后端

def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """x: [T] 单声道 或 [T, C] 多声道 -> 重采样到 sr_to。

    优先用 torchaudio.transforms.Resample，导入或调用失败时回退 scipy.signal.resample_poly。
    注意: 本项目 **不使用 librosa**，librosa 只出现在实验一~四的教学脚本里。
    """
    if sr_from == sr_to:
        return x
    try:
        import torch
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr_from, sr_to)
        if x.ndim == 1:
            out = resampler(torch.from_numpy(np.ascontiguousarray(x)))
            return out.numpy().astype(x.dtype, copy=False)
        # 多声道: torchaudio 期望 [C, T]
        out = resampler(torch.from_numpy(np.ascontiguousarray(x.T)))
        return np.ascontiguousarray(out.numpy().T, dtype=x.dtype)
    except Exception:  # noqa: BLE001
        pass
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(x, up, down, axis=0).astype(x.dtype, copy=False)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- 工具

def speaker_of(filename: str) -> str:
    """p226_001.wav -> p226"""
    stem = Path(filename).stem
    return stem.split("_")[0]


def natural_key(path: str):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]


# ---------------------------------------------------------------- 核心

def extract_and_resample(
    zip_path: Path,
    out_dir: Path,
    sr_target: int,
    workers: int,
    overwrite: bool,
) -> int:
    """把 zip 里的 wav 解压并(必要时)重采样到 out_dir，返回文件数。"""
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到 {zip_path}，请先运行 download.py")

    out_dir.mkdir(parents=True, exist_ok=True)
    done = len(list(out_dir.glob("*.wav")))
    if done > 0 and not overwrite:
        log(f"  已存在 {done} 个 wav，跳过 {zip_path.name} (如需重来加 --overwrite)")
        return done

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".wav")]
    names.sort(key=natural_key)
    log(f"  {zip_path.name}: {len(names)} 个 wav")

    def handle(name: str) -> str:
        dst = out_dir / Path(name).name
        if dst.exists() and not overwrite:
            return dst.name
        # 每个工作线程独立打开一次 ZipFile: 共享句柄在多线程下会因 seek 竞争读坏数据
        with zipfile.ZipFile(zip_path) as zf_local:
            with zf_local.open(name) as member:
                raw = member.read()
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if sr != sr_target:
            audio = _resample(audio, sr, sr_target)
        audio = np.clip(audio, -1.0, 1.0)  # 重采样可能轻微过冲, 写入 PCM_16 前先截断
        sf.write(str(dst), audio, sr_target, subtype="PCM_16")
        return dst.name

    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(handle, n) for n in names]
        for fut in as_completed(futures):
            fut.result()
            processed += 1
            if processed % 1000 == 0:
                log(f"    已处理 {processed}/{len(names)}")
    return len(names)


def write_manifest(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["utt_id", "clean_path", "noisy_path", "n_samples", "duration"])
        writer.writeheader()
        writer.writerows(rows)


def build_split(
    clean_dir: Path,
    noisy_dir: Path,
    keep_speakers: set[str] | None,
) -> list[dict]:
    clean_files = sorted(clean_dir.glob("*.wav"), key=lambda p: natural_key(p.name))
    rows: list[dict] = []
    for clean_path in clean_files:
        if keep_speakers is not None and speaker_of(clean_path.name) not in keep_speakers:
            continue
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            log(f"  [警告] 缺少对应的带噪文件: {noisy_path.name}，跳过")
            continue
        info = sf.info(clean_path)
        rows.append({
            "utt_id": clean_path.stem,
            # 存绝对路径: 后续 train/evaluate 无论在哪个目录运行都能读到
            "clean_path": clean_path.resolve().as_posix(),
            "noisy_path": noisy_path.resolve().as_posix(),
            "n_samples": int(info.frames),
            "duration": round(float(info.frames) / info.samplerate, 3),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 VoiceBank-DEMAND 训练数据")
    add_config_args(ap)
    ap.add_argument("--downloads", type=Path, default=None, help="覆盖 config 的 paths.downloads")
    ap.add_argument("--data", type=Path, default=None, help="覆盖 config 的 paths.data")
    ap.add_argument("--sr", type=int, default=None, choices=(8000, 16000, 24000, 48000), help="目标采样率")
    ap.add_argument("--spk", type=int, choices=(28, 56), default=None, help="训练集说话人版本")
    ap.add_argument("--val-spks", type=str, default=None,
                    help="用作验证集的说话人(逗号分隔)，与多数论文一致；不存在时自动回退随机划分")
    ap.add_argument("--val-ratio", type=float, default=None, help="按说话人随机划分验证集的比例，>0 时覆盖 --val-spks")
    ap.add_argument("--workers", type=int, default=None, help="并发线程数")
    ap.add_argument("--overwrite", action="store_true", help="忽略已存在的 wav，全部重做")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)          # 路径全部来自 YAML(paths 段)

    sr = int(pick(args.sr, cfg.data.sr))
    spk = int(pick(args.spk, cfg.data.spk))
    val_spks = pick(args.val_spks, cfg.data.val_spks)
    val_ratio = float(pick(args.val_ratio, cfg.data.val_ratio))
    workers = int(pick(args.workers, cfg.data.workers))
    seed = int(pick(args.seed, cfg.data.seed))
    overwrite = bool(args.overwrite or cfg.data.overwrite)

    # 命令行若覆盖了根目录, 子目录名仍按 config 的相对关系重算
    downloads = Path(args.downloads) if args.downloads is not None else P.downloads
    data_root = Path(args.data) if args.data is not None else P.data
    wav_root = data_root / P.wav.relative_to(P.data)
    manifest_root = data_root / P.manifest.relative_to(P.data)

    def zip_of(key: str) -> Path:
        return P.zip[key] if downloads == P.downloads else downloads / P.zip[key].name

    # 子集/通道目录名同样来自 config(paths.splits / paths.kinds)
    split_names = [str(s) for s in cfg.paths.splits]        # train, valid, test
    kind_names = [str(k) for k in cfg.paths.kinds]          # clean, noisy
    s_train, s_valid, s_test = (split_names + ["train", "valid", "test"])[:3]
    k_clean, k_noisy = (kind_names + ["clean", "noisy"])[:2]

    def wav_dir(split: str, kind: str) -> Path:
        return wav_root / split / kind

    random.seed(seed)

    log(f"采样率 {sr} Hz，输出 {wav_root}  (zip 目录 {downloads})")

    # ---- 1. 解压 + 重采样
    # 打印实际生效的重采样后端, 避免"以为在用 librosa"之类的误会
    try:
        import torchaudio  # noqa: F401
        log("重采样后端: torchaudio.transforms.Resample (失败会回退 scipy.signal.resample_poly)")
    except Exception:  # noqa: BLE001
        log("重采样后端: scipy.signal.resample_poly (torchaudio 不可用)")
    jobs = [
        (zip_of("clean_train"), wav_dir(s_train, k_clean)),
        (zip_of("noisy_train"), wav_dir(s_train, k_noisy)),
        (zip_of("clean_test"), wav_dir(s_test, k_clean)),
        (zip_of("noisy_test"), wav_dir(s_test, k_noisy)),
    ]
    missing = []
    for zip_path, out_dir in jobs:
        if not zip_path.exists():
            missing.append(zip_path.name)
            log(f"  跳过 {zip_path.name}（没下载，补跑 download.py 即可）")
            continue
        extract_and_resample(zip_path, out_dir, sr, workers, overwrite)
    if len(missing) == len(jobs):
        log("[错误] 一个 zip 都没有，请先运行 download.py")
        return

    # ---- 2. 划分验证集
    train_clean_dir = wav_dir(s_train, k_clean)
    all_speakers = sorted({speaker_of(p.name) for p in train_clean_dir.glob("*.wav")})

    if not all_speakers:
        log("  未发现训练集 wav，跳过验证集划分（只有测试集时只能评估，不能训练）")
        val_speakers, train_speakers = set(), set()
    elif val_ratio > 0:
        k = max(1, int(round(len(all_speakers) * val_ratio)))
        val_speakers = set(random.sample(all_speakers, k))
        train_speakers = set(all_speakers) - val_speakers
    else:
        asked = [s.strip() for s in val_spks.split(",") if s.strip()]
        val_speakers = {s for s in asked if s in all_speakers}
        missing = [s for s in asked if s not in all_speakers]
        if missing:
            log(f"  指定的说话人 {missing} 不在训练集中，自动随机选取代替")
            pool = [s for s in all_speakers if s not in val_speakers]
            val_speakers |= set(random.sample(pool, min(len(missing), len(pool))))
        train_speakers = set(all_speakers) - val_speakers

    log(f"  说话人: {len(all_speakers)} 人 -> 训练 {len(train_speakers)}，验证 {sorted(val_speakers)}")

    # 验证集的 wav 直接搬到 data/wav16k/valid/，保持目录干净
    if val_speakers:
        for kind in (k_clean, k_noisy):
            src_dir = wav_dir(s_train, kind)
            dst_dir = wav_dir(s_valid, kind)
            dst_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for wav in sorted(src_dir.glob("*.wav")):
                if speaker_of(wav.name) in val_speakers:
                    target = dst_dir / wav.name
                    if target.exists() and not overwrite:
                        continue
                    shutil.move(str(wav), str(target))
                    moved += 1
            log(f"  {kind}: 移入验证集 {moved} 条")

    # ---- 3. 生成清单(文件名与后缀也来自 config)
    suffix = str(cfg.paths.manifest_suffix)
    rows_train = build_split(wav_dir(s_train, k_clean), wav_dir(s_train, k_noisy), None)
    rows_valid = build_split(wav_dir(s_valid, k_clean), wav_dir(s_valid, k_noisy), None)
    rows_test = build_split(wav_dir(s_test, k_clean), wav_dir(s_test, k_noisy), None)

    write_manifest(rows_train, manifest_root / f"{s_train}{suffix}")
    write_manifest(rows_valid, manifest_root / f"{s_valid}{suffix}")
    write_manifest(rows_test, manifest_root / f"{s_test}{suffix}")

    def dur(rows_: list[dict]) -> float:
        return sum(r["duration"] for r in rows_)

    summary = {
        "sample_rate": sr,
        "trainset_speakers": spk,
        "train": {"num_utterances": len(rows_train), "hours": round(dur(rows_train) / 3600, 2)},
        "valid": {"num_utterances": len(rows_valid), "hours": round(dur(rows_valid) / 3600, 2)},
        "test": {"num_utterances": len(rows_test), "hours": round(dur(rows_test) / 3600, 2)},
        "val_speakers": sorted(val_speakers),
        "paths": {"data": str(data_root), "wav": str(wav_root), "manifest": str(manifest_root)},
    }
    (manifest_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n" + "=" * 56)
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    log("\n下一步: python dataset.py     # 检查数据加载(路径读 config)")
    log("        python train_demo.py  # 跑一个基线模型")


if __name__ == "__main__":
    main()
