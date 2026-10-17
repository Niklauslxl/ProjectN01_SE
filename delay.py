#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双讲(监听自身外放)延迟估计 —— 纯标准库 + numpy, 不需要任何额外依赖。

原理: 广义互相关加相位变换 (GCC-PHAT)。把两条波形做 FFT, 只保留互谱的相位,
      再逆变换, 峰值位置就是延迟(采样点)。

为什么需要它:
  用扬声器外放、用麦克风录音做"实时"演示时, 麦克风会先录到扬声器里传出来的
  增强后语音, 再录到人声。后端对这个"已经增强过一遍"的信号再增强一次, 于是
  出现了二次降噪的金属感。只要估计出这段回声的延迟, 就能在播放时把它抵消掉:
     播放信号 = 原始增强输出 - 延迟后的增强输出 * 增益
  人声先到(几乎无延迟), 反馈后到, 错开时间就不会互相抵消。

用法:
  1. 先跑 serve.py, 用一段正常说话测出延迟(建议 3 秒以上, 环境安静)
  2. python delay.py --ref 参考音频.wav --rec 录音.wav
  3. 把得到的 delay_ms / gain 填进 web/app.js 顶部的 FEEDBACK 配置

  也可以直接对着内置麦克风说一句话, 用扬声器放一段已知音频来测。
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        width = f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if width != 2:
        raise SystemExit(f"只支持 16bit PCM wav, 当前 {width*8} bit（先用 prepare/serve 转一下）")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def gcc_phat(ref: np.ndarray, rec: np.ndarray, sr: int, max_ms: float = 500.0) -> tuple[float, float]:
    """返回 (延迟毫秒, 归一化峰值)。正值表示 rec 比 ref 晚(回声)。

    推导: 若 rec[n] ≈ ref[n - D], 即录音比参考晚 D 个采样点, 循环互相关
    sum_n rec[n] * ref[n - k] 在 k = D 处最大。rfft 的解析频率网格恰好与
    循环相关一致, 所以 X * conj(R) 的 irfft 在索引 D 处取峰值, 直接用该
    索引即可 —— 不要再做 fftshift 式的搬移, 否则符号会反。
    """
    nfft = 1 << (int(ref.size + rec.size) - 1).bit_length()
    R = np.fft.rfft(ref, nfft)
    X = np.fft.rfft(rec, nfft)
    cross = X * np.conj(R)
    mag = np.abs(cross)
    cross = cross / np.maximum(mag, 1e-12)      # PHAT: 只留相位
    cc = np.fft.irfft(cross, nfft)

    lim = min(int(max_ms / 1000.0 * sr), nfft // 2 - 1)
    # 非负延迟(rec 晚)在 [0, lim]; 负延迟(rec 早)在 [nfft-lim, nfft)
    cand = np.concatenate([cc[: lim + 1], cc[nfft - lim:]])
    idx = int(np.argmax(cand))
    lag = idx if idx <= lim else idx - (2 * lim + 1) - (nfft - 2 * lim - 1)
    # 归一化峰值: 除以互相关序列的标准差, >6 基本就可信
    peak = float(cand[idx] / (np.std(cc) + 1e-12))
    return lag / sr * 1000.0, peak


def main() -> None:
    ap = argparse.ArgumentParser(description="估计扬声器->麦克风的回声延迟(GCC-PHAT)")
    ap.add_argument("--ref", type=Path, help="参考信号/播放的音频")
    ap.add_argument("--rec", type=Path, help="麦克风录到的音频")
    ap.add_argument("--max-ms", type=float, default=500.0)
    ap.add_argument("--self-test", action="store_true",
                    help="用合成信号验证算法本身是否正确, 与业务数据无关")
    args = ap.parse_args()

    if args.self_test:
        rng = np.random.default_rng(0)
        sr = 16000
        for true_ms in (137.0, 12.5, 45.0):
            true_lag = int(round(true_ms / 1000 * sr))
            ref = rng.standard_normal(2 * sr).astype(np.float64)
            rec = np.concatenate([np.zeros(true_lag), ref]) + 0.05 * rng.standard_normal(2 * sr + true_lag)
            est, pk = gcc_phat(ref, rec, sr, args.max_ms)
            print(f"  真实 {true_ms:6.1f} ms -> 估计 {est:6.1f} ms  (误差 {est-true_ms:+.2f} ms, 峰值 {pk:.1f})")
        return

    if not args.ref or not args.rec:
        ap.error("需要同时给出 --ref 和 --rec，或用 --self-test")

    ref, sr1 = read_wav(args.ref)
    rec, sr2 = read_wav(args.rec)
    if sr1 != sr2:
        raise SystemExit(f"两条音频采样率不同: {sr1} vs {sr2}")

    delay_ms, peak = gcc_phat(ref, rec, sr1, args.max_ms)
    print(f"采样率 {sr1} Hz  参考 {ref.size} 点  录音 {rec.size} 点")
    print(f"估计延迟: {delay_ms:.1f} ms   （GCC-PHAT 峰值 {peak:.1f}, 越大越可信）")
    print(f"可直接填进 web/app.js:  delayMs: {delay_ms:.0f},  gain: 0.9")


if __name__ == "__main__":
    sys.exit(main())
