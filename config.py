#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置中心: 所有路径与超参都从 YAML 读取, 命令行参数可再覆盖。

设计原则
--------
1. **代码里不出现任何具体路径**。目录结构( wav 放哪、manifest 叫什么、zip 叫什么 )
   全部由 config/*.yaml 描述, 换机器/换数据只改 YAML。
2. **优先级: 命令行 > --set > YAML > 内置默认**。所以老命令(
   `python train_demo.py --data ./data_real`) 依然有效, 只是默认值改从 YAML 取。
3. 相对路径一律以 `paths.root`(默认项目根目录) 为基准解析成绝对路径,
   这样从任何工作目录运行脚本都读到同一份数据。

用法
----
    from config import load_config, resolve_paths, add_config_args

    ap = argparse.ArgumentParser()
    add_config_args(ap)                      # 注入 --config / --set
    ...                                      # 其它参数 default 一律写 None
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)  # 配置对象(支持属性访问)
    P = resolve_paths(cfg)                    # 解析好的绝对路径

    root = P.data                             # 数据根目录
    csv  = P.csv["train"]                     # manifest/train.csv

命令行覆盖示例
--------------
    python train_demo.py --config quick
    python train_demo.py --set train.epochs=5 --set train.lr=3e-4
    VBD_CONFIG=config/data48k.yaml python prepare.py
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("缺少 PyYAML, 请先安装: pip install pyyaml")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_NAME = "default.yaml"
ENV_VAR = "VBD_CONFIG"

# ------------------------------------------------------------------ 内置默认
# 即使一个 YAML 都没有, 脚本也能跑起来(等价于重构前的硬编码值)。

DEFAULTS: dict[str, Any] = {
    "paths": {
        # 所有相对路径的基准; "." = 本文件所在目录(项目根)
        "root": ".",
        "downloads": "downloads",      # download.py 存放 zip 的目录
        "data": "data_real",           # prepare.py 输出根目录(wav + manifest 都在这下面)
        "runs": "runs/vbd_base",       # train_demo.py 存放 checkpoint 的目录
        "enhanced": "enhanced",        # evaluate.py 导出试听音频的目录
        # ---- 目录命名规则(支持 {sr}=kHz 数、{spk}=说话人数 占位符)
        "wav_dir": "wav{sr}k",
        "manifest_dir": "manifest",
        "manifest_suffix": ".csv",
        "zip": {                        # download.py 下载出的文件名
            "clean_train": "clean_trainset_{spk}spk_wav.zip",
            "noisy_train": "noisy_trainset_{spk}spk_wav.zip",
            "clean_test": "clean_testset_wav.zip",
            "noisy_test": "noisy_testset_wav.zip",
        },
        "splits": ["train", "valid", "test"],   # 子集目录名
        "kinds": ["clean", "noisy"],            # 每个子集下的两个子目录
    },
    "download": {
        "base_url": "https://datashare.ed.ac.uk/bitstreams",
        "spk": 28,
        "only_test": False,
        "proxy": None,
        "retries": 5,
        "insecure": False,
        # name -> [bitstream_id, 近似大小MB, 说明]
        "files": {
            "clean_trainset_28spk_wav": ["245452b6-6235-44b6-a6f9-e7eb19797769", 2380.0, "28说话人 干净训练集"],
            "noisy_trainset_28spk_wav": ["ecb5a102-bb00-46d3-8af5-40c79823b837", 2700.0, "28说话人 带噪训练集"],
            "clean_trainset_56spk_wav": ["386ac199-5d34-402f-8112-2a123577dd57", 4548.0, "56说话人 干净训练集"],
            "noisy_trainset_56spk_wav": ["60506429-e89f-4ae5-a28f-da63238d11cb", 5366.0, "56说话人 带噪训练集"],
            "clean_testset_wav": ["dec213d3-bf57-4777-9663-c24bdce92d5e", 147.2, "干净测试集"],
            "noisy_testset_wav": ["13c1bfbf-14a6-41db-9b41-8f7310f01ad5", 162.7, "带噪测试集"],
        },
    },
    "data": {
        "sr": 16000,
        "spk": 28,
        "val_spks": "p282,p287",
        "val_ratio": 0.0,
        "workers": 8,
        "overwrite": False,
        "seed": 42,
    },
    "dataset": {
        "segment": 3.0,
        "batch_size": 16,
        "num_workers": 4,
        "remix": True,
        "snr_range": [0.0, 20.0],
        "random_gain": [0.7, 1.0],
        "repeat_if_short": True,
    },
    "model": {
        "n_fft": 512,
        "hop": 128,
        "win": 512,
        "channels": [16, 32, 64, 128],
    },
    "loss": {
        "w_snr": 0.8,
        "w_mag": 0.2,
        "clip_grad": 5.0,
    },
    "train": {
        "epochs": 40,
        "batch_size": 16,
        "lr": 1e-3,
        "segment": 3.0,
        "num_workers": 4,
        "device": "auto",
        "seed": 1234,
        "resume": None,
        "max_steps": 0,
        "patience": 0,
        "save_each": False,
    },
    "eval": {
        "ckpt": None,          # None -> 用 <paths.runs>/best.pt
        "limit": 0,
        "dump": None,
        "sr": 16000,
        "device": "auto",
    },
}


# ------------------------------------------------------------------ 数据结构

class AttrDict(dict):
    """支持属性访问的 dict(`cfg.train.epochs` 等价于 `cfg["train"]["epochs"]`)。"""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attr(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并, override 里的同名键覆盖 base, 未提到的键保留。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(raw: str) -> Any:
    """`--set k=v` 的 v 先按 YAML 解析(能认出 true/null/数字/列表), 失败则当字符串。

    YAML 1.1 不认 `3e-4` 这种科学计数法(会当成字符串), 所以再兜一层数值转换。
    """
    try:
        value = yaml.safe_load(raw)
    except Exception:  # noqa: BLE001
        return raw
    if isinstance(value, str):
        try:
            return float(value) if ("." in value or "e" in value.lower()) else int(value)
        except ValueError:
            return value
    return value


def _flatten_overrides(overrides: Iterable[str] | None) -> list[str]:
    """--set 用了 action="append" + nargs="*", 这里把嵌套列表摊平成一维。"""
    items: list[str] = []
    for group in overrides or []:
        if isinstance(group, str):
            items.append(group)
        elif group:
            items.extend(str(g) for g in group)
    return items


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise SystemExit(f"--set {dotted_key}: 父节点 {part} 不是字典")
    node[parts[-1]] = value


# ------------------------------------------------------------------ 加载

def resolve_config_path(name: str | os.PathLike | None) -> Path | None:
    """把 --config 的值解析成实际的 YAML 路径。

    支持: 完整路径 / config/xxx.yaml / 简写 xxx(自动补全成 config/xxx.yaml)。
    没传时看环境变量 VBD_CONFIG, 最后回落到 config/default.yaml。
    """
    if name is None:
        name = os.environ.get(ENV_VAR) or DEFAULT_CONFIG_NAME
    p = Path(str(name))
    candidates = [p]
    if not p.is_absolute():
        candidates += [CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml", PROJECT_ROOT / p]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    # 一个都没找到: 只有显式指定时才报错, 默认名缺失则安静回落内置默认
    if name != DEFAULT_CONFIG_NAME:
        raise SystemExit(f"找不到配置文件: {name}（已尝试 {', '.join(str(c) for c in candidates)}）")
    return None


def load_config(
    path: str | os.PathLike | None = None,
    overrides: Iterable[str] | None = None,
    defaults: dict | None = None,
) -> AttrDict:
    """内置默认 -> YAML -> --set, 逐层覆盖后返回配置对象。"""
    cfg = copy.deepcopy(defaults if defaults is not None else DEFAULTS)
    yaml_path = resolve_config_path(path)
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in _flatten_overrides(overrides):
        if "=" not in item:
            raise SystemExit(f"--set 格式应为 key.path=value, 收到: {item}")
        key, _, value = item.partition("=")
        set_nested(cfg, key.strip(), _parse_value(value.strip()))
    return _to_attr(cfg)


# ------------------------------------------------------------------ 路径解析

def resolve_paths(cfg: AttrDict) -> AttrDict:
    """把 paths 段展开成一组可用的绝对路径。

    返回键:
        root / downloads / data / wav / manifest / runs / enhanced
        csv        : {split: manifest/<split>.csv}
        zip        : {clean_train, noisy_train, clean_test, noisy_test: 绝对路径}
        wav_split  : {split: {kind: 绝对路径}}
    """
    root = Path(str(cfg.paths.root))
    root = root if root.is_absolute() else (PROJECT_ROOT / root).resolve()

    def sub(value: Any) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else (root / p)

    sr = int(cfg.data.sr)
    spk = int(cfg.data.spk)
    fmt = {"sr": sr // 1000, "spk": spk}

    data = sub(cfg.paths.data)
    wav = sub(cfg.paths.data) / str(cfg.paths.wav_dir).format(**fmt)
    manifest = sub(cfg.paths.data) / str(cfg.paths.manifest_dir)
    downloads = sub(cfg.paths.downloads)

    splits = [str(s) for s in cfg.paths.splits]
    kinds = [str(k) for k in cfg.paths.kinds]
    suffix = str(cfg.paths.manifest_suffix)

    paths = AttrDict(
        root=root,
        downloads=downloads,
        data=data,
        wav=wav,
        manifest=manifest,
        runs=sub(cfg.paths.runs),
        enhanced=sub(cfg.paths.enhanced),
        csv={s: manifest / f"{s}{suffix}" for s in splits},
        zip={k: downloads / str(tpl).format(**fmt) for k, tpl in cfg.paths.zip.items()},
        wav_split={s: {k: wav / s / k for k in kinds} for s in splits},
    )
    return paths


def add_config_args(ap) -> None:
    """给任意脚本注入 --config / --set 两个通用参数。"""
    ap.add_argument("--config", default=None,
                    help=f"配置文件: 完整路径 / config/xxx.yaml / 简写 xxx（默认 {DEFAULT_CONFIG_NAME}）")
    # action="append" 而非默认覆盖: 否则写多个 --set 时只有最后一个生效(argparse 的坑)
    ap.add_argument("--set", action="append", nargs="*", default=[], metavar="KEY=VALUE",
                    help="临时覆盖配置项, 如 --set train.epochs=5 --set paths.data=./data48k")


def pick(cli_value: Any, cfg_value: Any) -> Any:
    """命令行没给(None)就用配置里的值。"""
    return cli_value if cli_value is not None else cfg_value


def main() -> None:
    """自检: 打印当前配置的解析结果。python config.py [--config xxx]"""
    import argparse

    import json

    ap = argparse.ArgumentParser(description="查看当前生效的配置")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)

    print("配置文件:", resolve_config_path(args.config) or "(未找到, 使用内置默认)")
    print("\n[解析后的路径]")
    for key in ("root", "downloads", "data", "wav", "manifest", "runs", "enhanced"):
        print(f"  {key:<10} {P[key]}")
    print("  csv        ", {k: str(v) for k, v in P.csv.items()})
    print("  zip        ", {k: str(v) for k, v in P.zip.items()})
    print("  wav_split  ", {s: {k: str(v) for k, v in d.items()} for s, d in P.wav_split.items()})
    print("\n[关键超参]")
    print("  data.sr =", cfg.data.sr, " data.spk =", cfg.data.spk, " val_spks =", cfg.data.val_spks)
    print("  train   =", json.dumps({
        k: cfg.train[k] for k in ("epochs", "batch_size", "lr", "segment", "patience", "save_each")}))
    print("  model   =", json.dumps({k: cfg.model[k] for k in ("n_fft", "hop", "win", "channels")}))


if __name__ == "__main__":
    main()
