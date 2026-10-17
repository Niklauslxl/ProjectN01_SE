#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地语音增强演示服务（完全离线，只监听 127.0.0.1）

    python serve.py                      # 打开 http://127.0.0.1:8000
    python serve.py --port 8080
    python serve.py --ckpt runs/real_base/best.pt

浏览器里可以：麦克风实时录音 / 上传音频文件 -> 看增强前后语谱图 -> 对比试听 -> 下载。

只用标准库 http.server + 已有的 torch/numpy/soundfile，不联网、不装新包。
前端把音频统一转成 16 kHz 单声道 WAV 的 base64 发过来，后端推理完再把
增强后的 WAV 和两张语谱图矩阵回传，语谱图由前端 canvas 绘制。

注意: 实时录音没有"干净参考", 所以算不了 SI-SNR / STOI / PESQ。
这里改用两个无参考指标:
  - noise_reduction_db : 噪声帧(输入能量最低 10% 的帧)能量下降了多少 dB, 越大越好
  - speech_change_db   : 语音帧(能量最高 50% 的帧)能量变化, 接近 0 最好,
                         明显为负说明"过度抑制"(把人声也削掉了)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from config import load_config, resolve_paths
from train_demo import MaskUNet

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

MAX_SECONDS = 60.0        # 单次处理上限，防止浏览器等太久
TARGET_SR = 16000
SPEC_BINS = 128           # 语谱图频率方向降采样后的行数
SPEC_FRAMES = 360         # 时间方向最多多少列

STATE: dict = {}


# ------------------------------------------------------------------ 模型

def load_model(cfg, ckpt_override: str | None = None):
    P = resolve_paths(cfg)
    ckpt_path = Path(ckpt_override) if ckpt_override else Path(cfg.eval.ckpt) if cfg.eval.ckpt else P.runs / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"找不到权重 {ckpt_path}（用 --ckpt 指定，或改 config 的 eval.ckpt）")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MaskUNet(
        n_fft=int(cfg.model.n_fft),
        hop=int(cfg.model.hop),
        win=int(cfg.model.win),
        channels=tuple(cfg.model.channels),
    )
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    # device: "auto" 时优先用 GPU；CPU 机器上自动退回 CPU
    dev_arg = str(cfg.eval.device) if getattr(cfg.eval, "device", None) else "auto"
    device = torch.device(
        "cuda" if (dev_arg == "auto" and torch.cuda.is_available())
        else (dev_arg if dev_arg != "auto" else "cpu"))
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "ckpt": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "best_si_snri": ckpt.get("best"),
        "params": n_params,
        "n_fft": int(cfg.model.n_fft),
        "hop": int(cfg.model.hop),
        "sr": TARGET_SR,
        "device": str(device),
    }
    print(f"已加载权重: {ckpt_path}")
    print(f"  epoch={info['epoch']}  best SI-SNRi={info['best_si_snri']}  "
          f"参数量={n_params/1e6:.3f}M  设备={device}")
    return model, info


# ------------------------------------------------------------------ 推理

@torch.no_grad()
def enhance_wave(x: torch.Tensor, sr: int, model) -> torch.Tensor:
    """长音频分块推理: 4 秒块、50% 重叠、汉宁窗加权叠加，避免整句塞进显存/内存。"""
    n = x.numel()
    chunk = 4 * sr
    step = chunk // 2
    if n <= chunk:
        return model(x.unsqueeze(0)).squeeze(0)

    win = torch.hann_window(chunk, device=x.device)
    out = torch.zeros(n, device=x.device)
    wsum = torch.zeros(n, device=x.device)
    start = 0
    while start < n:
        end = min(start + chunk, n)
        length = end - start
        seg = x[start:end]
        if seg.numel() < chunk:
            seg = F.pad(seg, (0, chunk - seg.numel()))
        y = model(seg.unsqueeze(0)).squeeze(0)[:length]

        w = torch.ones(length, device=x.device)
        if start != 0:                      # 非首块: 左半边做淡入
            w[: min(step, length)] = win[: min(step, length)]
        if end != n:                        # 非尾块: 右半边做淡出
            tail = length - step
            if tail > 0:
                w[step:] = win[step:length]
        out[start:end] += y * w
        wsum[start:end] += w
        start += step
    return out / wsum.clamp(min=1e-6)


def spectrogram(x: torch.Tensor, n_fft: int, hop: int) -> list[list[int]]:
    """返回 [频率行][时间列] 的 0~255 灰度矩阵，低频在第 0 行。"""
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, n_fft, window=window, return_complex=True).abs()
    db = 20 * torch.log10(spec.clamp(min=1e-8))              # [F, T]
    cols = min(SPEC_FRAMES, db.shape[-1])
    pooled = F.adaptive_avg_pool2d(db[None, None], (SPEC_BINS, cols))[0, 0]
    arr = pooled.numpy()
    arr = (np.clip((arr + 80.0) / 80.0, 0.0, 1.0) * 255).astype(np.uint8)
    return arr.tolist()


def frame_energy(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, n_fft, window=window, return_complex=True)
    return spec.abs().pow(2).mean(dim=0)


def _safe_ratio_db(num: float, den: float, floor: float = 1e-10) -> float:
    """10*log10(num/den)，并保证结果是有限数值。

    为什么必须做这层保护：录音的开头/结尾常有纯静音段，此时帧能量为 0，
    `10*log10(0/0)` 会得到 -inf / nan。Python 的 json.dumps 会把 inf/nan
    原样写成 `-Infinity` / `NaN`，而这两个是 **JSON 标准不允许的字面量**，
    浏览器 JSON.parse 读到 `-I` 就抛出
        No number after minus sign in JSON at position N
    报错位置落在响应中段（前面是大块语谱图矩阵），很容易被误判成"数据截断"。
    这里统一把结果夹到 [-60, 60] dB 的合理区间内。
    """
    if den <= 0 or num <= 0:
        return -60.0
    return max(-60.0, min(60.0, 10.0 * math.log10(max(num, floor) / max(den, floor))))


def compute_metrics(noisy: torch.Tensor, est: torch.Tensor, n_fft: int, hop: int) -> dict:
    e_in = frame_energy(noisy, n_fft, hop)
    e_out = frame_energy(est, n_fft, hop)

    k_noise = max(1, int(e_in.numel() * 0.10))
    noise_idx = torch.topk(e_in, k_noise, largest=False).indices
    nr_db = _safe_ratio_db(float(e_in[noise_idx].mean()), float(e_out[noise_idx].mean()))

    k_speech = max(1, int(e_in.numel() * 0.50))
    speech_idx = torch.topk(e_in, k_speech).indices
    sc_db = _safe_ratio_db(float(e_out[speech_idx].mean()), float(e_in[speech_idx].mean()))

    return {
        "noise_reduction_db": round(nr_db, 2),
        "speech_change_db": round(sc_db, 2),
    }


def wav_to_b64(x: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process(audio_b64: str) -> dict:
    model = STATE["model"]
    n_fft, hop = STATE["n_fft"], STATE["hop"]

    if not audio_b64:
        raise ValueError("请求里没有 audio 字段（前端可能没选音频就点了增强）")
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"base64 解码失败，数据可能被截断（共 {len(audio_b64)} 字符）: {exc}")
    print(f"    收到音频 {len(raw)/1024:.1f} KB")

    try:
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ValueError(f"不是能识别的音频格式（前端应转成 16k 单声道 WAV）: {exc}")
    if x.ndim > 1:
        x = x.mean(axis=1)

    duration = len(x) / sr
    print(f"    解码完成 {duration:.2f}s @ {sr} Hz，开始推理…")
    if duration > MAX_SECONDS:
        raise ValueError(f"音频太长（{duration:.1f}s），上限 {MAX_SECONDS:.0f}s")
    if duration < 0.05:
        raise ValueError("音频太短")

    # 统一到 16 kHz（前端已经转过，这里只是兜底）
    if sr != TARGET_SR:
        new_len = max(1, int(round(len(x) * TARGET_SR / sr)))
        y = F.interpolate(torch.from_numpy(x)[None, None], size=new_len,
                          mode="linear", align_corners=False)
        x = y[0, 0].numpy()
        sr = TARGET_SR

    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    device = STATE.get("device", torch.device("cpu"))
    t0 = time.time()
    est = enhance_wave(wav.to(device), sr, model).cpu()
    cost = time.time() - t0
    print(f"    推理完成 {cost*1000:.0f} ms（RTF {cost/max(duration, 1e-6):.3f}），正在生成语谱图…")

    est_np = est.numpy()
    out = {
        "sr": sr,
        "duration": round(duration, 3),
        "audio": wav_to_b64(est_np, sr),
        "spec_before": spectrogram(wav, n_fft, hop),
        "spec_after": spectrogram(est, n_fft, hop),
        "metrics": compute_metrics(wav, est, n_fft, hop),
        "process_ms": int(cost * 1000),
        "rtf": round(cost / max(duration, 1e-6), 4),
        "model": STATE["info"],
    }
    n_cols = len(out["spec_before"][0])
    print(f"    语谱图 {len(out['spec_before'])}x{n_cols}，"
          f"返回 {len(out['audio'])/1024:.1f} KB base64")
    return out


# ------------------------------------------------------------------ HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceBankDemo"

    def _headers(self, ctype: str, length: int) -> None:
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self._headers(ctype, len(body))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # 浏览器按了停止/刷新就会断连；这不是错误，别把堆栈糊到终端上
            print("    （客户端提前断开连接，响应已丢弃）")

    def _send_json(self, code: int, obj: dict) -> None:
        # allow_nan=False: 让 json.dumps 在遇到 inf/nan 时**直接报错**，而不是
        # 悄悄写出 `-Infinity` / `NaN`。这两个词在 JSON 标准里非法，浏览器
        # JSON.parse 会以 "No number after minus sign" 之类的怪错失败，排查
        # 起来非常费劲。宁可在服务端明确失败。
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._headers("text/plain", 0)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            file_path = WEB_DIR / "index.html"
        elif path in ("/app.js", "/style.css"):
            file_path = WEB_DIR / path.lstrip("/")
        elif path == "/api/info":
            self._send_json(200, {"ok": True, "model": STATE["info"], "max_seconds": MAX_SECONDS})
            return
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        if not file_path.exists():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}.get(file_path.suffix, "application/octet-stream")
        self._send(200, file_path.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/enhance":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        t0 = time.time()
        length = int(self.headers.get("Content-Length", 0) or 0)
        print(f"[{time.strftime('%H:%M:%S')}] POST /api/enhance  "
              f"Content-Length={length} ({length/1024:.1f} KB)  "
              f"来自 {self.address_string()}")
        if length <= 0:
            self._send_json(400, {"ok": False, "error": "请求体为空（Content-Length 缺失）"})
            return
        print(f"    Content-Type: {self.headers.get('Content-Type')}  "
              f"Expect: {self.headers.get('Expect')}")
        try:
            body = self.rfile.read(length)
            print(f"    实际读到 {len(body)} 字节")
            if len(body) != length:
                raise ValueError(f"请求体不完整：只收到 {len(body)}/{length} 字节")
            payload = json.loads(body.decode("utf-8"))
            print(f"    JSON 解析成功，audio 字段 {len(payload.get('audio',''))} 字符")
            result = process(payload.get("audio", ""))
            result["ok"] = True
            print(f"    完成，总计 {time.time()-t0:.2f}s，已返回结果\n")
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            # 把学生能看懂的提示发到浏览器，同时终端也留一份，方便对着看
            print(f"    失败：{type(exc).__name__}: {exc}\n")
            self._send_json(400, {"ok": False, "error": str(exc)})

    def handle_one_request(self) -> None:
        """单个请求出错时不要悄悄关闭连接, 否则浏览器只会看到一个空的/截断的响应。"""
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            print("    （客户端提前断开连接）")
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001
            print(f"    !! 处理请求时出现未预期错误 {type(exc).__name__}: {exc}")
            self.close_connection = True

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    """允许复用地址，并把请求处理上的意外错误打印出来而不是静默断连。"""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        import traceback
        print(f"!! 处理 {client_address} 的请求时出错：")
        traceback.print_exc()


def port_in_use(host: str, port: int) -> bool:
    """检测端口是否已经被占用（Windows 上不需要 SO_REUSEADDR 就能连上）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def check_port_free(host: str, port: int) -> None:
    """启动前先确认端口空闲。

    这条检查是针对一个很常见的坑：上次的 serve.py 没关掉（窗口还在、或者
    被最小化了），新的又启动起来。两个进程抢同一个端口时，Windows 上后启动
    的那个不一定报错，浏览器却可能把请求发给**上一个旧进程** —— 于是你看到
    "终端没有日志 + 网页报各种奇怪错误"，还以为是代码坏了。
    """
    if not port_in_use(host, port):
        return
    raise SystemExit(
        f"\n端口 {port} 已经被占用 —— 八成就有一个 serve.py 还在跑。\n\n"
        f"先把它关掉再启动新的：\n"
        f"  1) 找到那个正在跑的 serve.py 窗口，按 Ctrl+C\n"
        f"  2) 找不到窗口就在另一个终端执行：\n"
        f"       netstat -ano | findstr :{port}\n"
        f"    记下最后一列的 PID，然后：\n"
        f"       taskkill /F /PID <PID>\n"
        f"  3) 或者换个端口：python serve.py --port 8001\n\n"
        f"为什么必须这样做：两个进程抢同一个端口时，浏览器可能把请求发给旧的\n"
        f"那一个，那个旧进程运行的是老代码，你的新日志自然一条也不会出现。\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="本地语音增强演示服务")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", nargs="*", default=[])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ckpt", default=None, help="指定权重，默认用 config 的 eval.ckpt / runs/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    check_port_free(args.host, args.port)
    model, info = load_model(cfg, args.ckpt)
    STATE.update(model=model, info=info, n_fft=int(cfg.model.n_fft), hop=int(cfg.model.hop),
                 device=torch.device(info["device"]))

    url = f"http://{args.host}:{args.port}"
    print(f"\n服务已启动: {url}")
    print(f"本次进程 PID = {os.getpid()}（关服务就是在当前窗口按 Ctrl+C）")
    print("浏览器打开上面的地址即可使用。Ctrl+C 停止。")
    print("每次点击“开始增强”，这个窗口都会打印进度；没打印说明请求根本没到后端。\n")

    server = ThreadingHTTPServerV6((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
