#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VoiceBank-DEMAND 数据集下载脚本

数据主页: https://datashare.ed.ac.uk/handle/10283/2791
出处: Valentini-Botinhao, C. (2017). Noisy speech database for training speech
      enhancement algorithms and TTS models. University of Edinburgh. DOI: 10.7488/ds/2117

特性:
  - 直连爱丁堡 DataShare，无需浏览器
  - HTTP Range 断点续传，断网重跑自动接着下
  - 失败自动重试 + 限速可控，适合国内网络
  - 下载后按 Content-Length 校验文件大小

保存目录、代理、重试次数、bitstream id 全部来自 config/*.yaml。

用法:
  python download.py                                   # 下载 28 说话人版(约 5.2GB)
  python download.py --spk 56                          # 下载 56 说话人版(约 10.5GB)
  python download.py --only-test                       # 只要测试集(约 310MB)
  python download.py --set download.proxy=http://127.0.0.1:7890
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import add_config_args, load_config, pick, resolve_paths

# 文件清单默认来自 config/*.yaml 的 download.files
# (name -> [bitstream_id, 近似大小MB, 说明])，id 来自 DataShare 主页 10283/2791
TRAIN_KEYS_28 = ["clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"]
TRAIN_KEYS_56 = ["clean_trainset_56spk_wav", "noisy_trainset_56spk_wav"]
TEST_KEYS = ["clean_testset_wav", "noisy_testset_wav"]

CHUNK = 1024 * 1024  # 1 MiB
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def build_opener(proxy: str | None = None, insecure: bool = False) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    # 个别网络环境下 DataShare 的证书链可能取不到，允许 insecure 关闭校验
    if insecure or os.environ.get("VBD_INSECURE") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def human(size_bytes: float) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def download_one(
    name: str,
    out_dir: Path,
    opener: urllib.request.OpenerDirector,
    files: dict,
    base_url: str,
    retries: int = 5,
) -> bool:
    bitstream_id, approx_mb, desc = files[name]
    url = f"{base_url}/{bitstream_id}/download"
    dst = out_dir / f"{name}.zip"
    if dst.is_dir():
        return True

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            pos = dst.stat().st_size if dst.exists() else 0
            if pos > 0:
                req.add_header("Range", f"bytes={pos}-")

            resp = opener.open(req, timeout=120)
            code = resp.getcode()
            total = int(resp.headers.get("Content-Length", 0))
            accepts_range = resp.headers.get("Accept-Ranges", "") == "bytes"

            if code == 200:
                # 服务器不支持续传或忽略 Range，从头开始
                if pos > 0:
                    print(f"[{name}] 服务器不支持断点续传，重新下载")
                pos = 0
                mode = "wb"
            elif code == 206:
                mode = "ab"
                if total:
                    total += pos  # 转成文件总大小
            elif code == 416:
                print(f"[{name}] 已下载完整，跳过")
                return True
            else:
                raise urllib.error.HTTPError(url, code, "unexpected status", resp.headers, None)

            if total == 0:
                total = int(approx_mb * 1024 * 1024)  # 兜底估算，仅用于进度显示

            print(f"[{name}] {desc}  目标 {human(total)}  已存在 {human(pos)}"
                  + ("" if accepts_range or code == 206 else "  (不支持续传)"))

            start = time.time()
            downloaded = pos
            last_print = 0.0
            with open(dst, mode) as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print > 1.0:
                        speed = (downloaded - pos) / (now - start + 1e-9)
                        pct = 100.0 * downloaded / max(total, 1)
                        sys.stdout.write(
                            f"\r[{name}] {human(downloaded)} / {human(total)}  {pct:5.1f}%  {human(speed)}/s   "
                        )
                        sys.stdout.flush()
                        last_print = now
            resp.close()

            actual = dst.stat().st_size
            if actual < total * 0.995:
                raise IOError(f"文件大小不足: {human(actual)} < {human(total)}")
            print(f"\r[{name}] 完成: {dst}  ({human(actual)})" + " " * 40)
            return True

        except Exception as exc:  # noqa: BLE001
            current = dst.stat().st_size if dst.exists() else 0
            print(f"\n[{name}] 第 {attempt}/{retries} 次尝试失败: {exc}  (已存 {human(current)})")
            if attempt >= retries:
                print(f"[{name}] 放弃。手动重跑本脚本可继续断点续传。")
                return False
            time.sleep(min(2 ** attempt, 30))
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 VoiceBank-DEMAND 数据集")
    add_config_args(ap)
    ap.add_argument("--out", type=Path, default=None, help="覆盖 config 的 paths.downloads")
    ap.add_argument("--spk", type=int, choices=(28, 56), default=None, help="训练集说话人数量")
    ap.add_argument("--only-test", action="store_true", default=None, help="只下载测试集")
    ap.add_argument("--proxy", type=str, default=None, help="HTTP 代理，例如 http://127.0.0.1:7890")
    ap.add_argument("--insecure", action="store_true", default=None, help="关闭 SSL 证书校验")
    ap.add_argument("--retries", type=int, default=None, help="每个文件的重试次数")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)

    out_dir = Path(args.out) if args.out is not None else P.downloads
    spk = int(pick(args.spk, cfg.download.spk))
    only_test = bool(args.only_test or cfg.download.only_test)
    proxy = pick(args.proxy, cfg.download.proxy)
    insecure = bool(args.insecure or cfg.download.insecure)
    retries = int(pick(args.retries, cfg.download.retries))
    files = {k: tuple(v) for k, v in cfg.download.files.items()}  # name -> (id, MB, 说明)
    base_url = str(cfg.download.base_url)

    keys = list(TEST_KEYS) if only_test else (TRAIN_KEYS_56 if spk == 56 else TRAIN_KEYS_28) + TEST_KEYS
    keys = [k for k in keys if k in files]  # 配置里没列的就跳过

    out_dir.mkdir(parents=True, exist_ok=True)
    opener = build_opener(proxy, insecure)

    print(f"保存目录: {out_dir.resolve()}")
    print(f"待下载 {len(keys)} 个文件，预计 "
          f"{human(sum(files[k][1] for k in keys) * 1024 * 1024)}\n")

    ok, fail = [], []
    for key in keys:
        if download_one(key, out_dir, opener, files, base_url, retries=retries):
            ok.append(key)
        else:
            fail.append(key)

    print("\n" + "=" * 56)
    print(f"成功 {len(ok)} 个，失败 {len(fail)} 个")
    if fail:
        print("失败列表: " + ", ".join(fail))
        sys.exit(1)
    print("下一步: python prepare.py")


if __name__ == "__main__":
    main()
