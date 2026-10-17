#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线训练脚本: 频域复数掩码 U-Net + SI-SNR 损失

模型很小(约 1.5M 参数), 单卡 3060 上几分钟就能跑完一个 epoch,
目的是验证整条数据链路没问题, 并给出一个可以对比的起点。

数据路径、超参全部来自 config/*.yaml, 命令行参数只是临时覆盖。

训练:
  python train_demo.py
  python train_demo.py --epochs 40 --batch-size 32 --lr 5e-4
  python train_demo.py --config quick                 # 冒烟预设
  python train_demo.py --set train.patience=8         # 连续 8 轮无提升就早停
  python train_demo.py --set train.save_each=true     # 每轮留一份快照(~40×6MB)
继续训练:
  python train_demo.py --resume runs/vbd_base/last.pt

设备/显卡:
  train.device 默认 "auto": 有 CUDA 就用 GPU 并自动开 AMP(混合精度) + pin_memory,
  没有就自动退回 CPU 且关闭 AMP。不需要为 GPU 准备另一份代码。
  换到显卡机器上训练:  python train_demo.py --config gpu
  强制用 CPU:          python train_demo.py --device cpu
  指定第二块卡:        python train_demo.py --device cuda:1
  注意: CPU 上训出的 last.pt 可以直接拿到 GPU 机器上 --resume 续训(脚本会把
  优化器状态一起搬到目标设备), 反过来 GPU -> CPU 也可以。

产物说明:
  runs/<out>/last.pt   每轮无条件覆盖, 用于 --resume 断点续训
  runs/<out>/best.pt   仅当 valid SI-SNRi 刷新最好成绩时才覆盖, 即"最优模型"
  runs/<out>/epoch_XXX.pt  只有加 --save-each 才生成
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import add_config_args, load_config, pick, resolve_paths
from dataset import VoiceBankDataset, collate

EPS = 1e-8


def _make_scaler(device_type: str, enabled: bool):
    """兼容 torch 新旧版本的 GradScaler 构造方式。"""
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)  # torch >= 2.3
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device_type: str, enabled: bool):
    """兼容 torch 新旧版本的 autocast 构造方式。"""
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)  # torch >= 2.4
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)


# ------------------------------------------------------------ 模型

def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 x 的空间尺寸对齐到 ref, 供跳跃连接使用。"""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """输入 [B,2,F,T] (实部+虚部), 输出复数掩码 [B,2,F,T]。"""

    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.e1 = ConvBlock(2, c1)
        self.e2 = ConvBlock(c1, c2)
        self.e3 = ConvBlock(c2, c3)
        self.e4 = ConvBlock(c3, c4)
        self.pool = nn.AvgPool2d((2, 2))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d3 = ConvBlock(c4 + c3, c3)
        self.d2 = ConvBlock(c3 + c2, c2)
        self.d1 = ConvBlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 2, (1, 1))
        nn.init.zeros_(self.out.weight)  # 初始掩码接近 0 -> 从"全抑制"开始, 收敛更稳
        nn.init.constant_(self.out.bias, 0.5)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.e4(self.pool(s3))
        # 频点数是奇数时(如 n_fft=512 -> 257 bin)上下采样无法整除对齐, 用插值补齐
        u = self.d3(torch.cat([_match_size(self.up(b), s3), s3], dim=1))
        u = self.d2(torch.cat([_match_size(self.up(u), s2), s2], dim=1))
        u = self.d1(torch.cat([_match_size(self.up(u), s1), s1], dim=1))
        return self.out(u)


class MaskUNet(nn.Module):
    def __init__(self, n_fft=512, hop=128, win=512, channels=(16, 32, 64, 128)):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.unet = TinyUNet(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, self.n_fft, self.hop, self.win, window=window, return_complex=True)
        feats = torch.stack([spec.real, spec.imag], dim=1)
        mask = self.unet(feats)
        mr, mi = mask[:, 0], mask[:, 1]
        est = torch.complex(mr * spec.real - mi * spec.imag, mr * spec.imag + mi * spec.real)
        return torch.istft(est, self.n_fft, self.hop, self.win, window=window, length=x.shape[-1])


# ------------------------------------------------------------ 损失

def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    return 20 * torch.log10(target.pow(2).sum(-1) / (noise.pow(2).sum(-1) + EPS) + EPS)


def combined_loss(
    est: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    n_fft: int = 512,
    hop: int = 128,
    w_snr: float = 0.8,
    w_mag: float = 0.2,
) -> torch.Tensor:
    snr_loss = -si_snr(est, clean).mean()
    window = torch.hann_window(n_fft, device=est.device)
    with torch.no_grad():
        ref_spec = torch.stft(clean, n_fft, hop, n_fft, window=window, return_complex=True).abs()
    est_spec = torch.stft(est, n_fft, hop, n_fft, window=window, return_complex=True).abs()
    mag_loss = F.l1_loss(est_spec, ref_spec)
    return w_snr * snr_loss / 20.0 + w_mag * mag_loss, snr_loss, mag_loss


# ------------------------------------------------------------ 训练 / 验证

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    before, after = [], []
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        est = model(noisy)
        after.append(si_snr(est, clean))
        before.append(si_snr(noisy, clean))
    return {
        "base_si_snr": float(torch.cat(before).mean()),
        "si_snr": float(torch.cat(after).mean()),
        "si_snri": float(torch.cat(after).mean() - torch.cat(before).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    add_config_args(ap)
    ap.add_argument("--data", type=Path, default=None, help="覆盖 config 的 paths.data")
    ap.add_argument("--out", type=Path, default=None, help="覆盖 config 的 paths.runs")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--segment", type=float, default=None, help="裁片长度(秒)")
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="每轮最多训几个 batch, 0=不限(冒烟测试用)")
    ap.add_argument("--patience", type=int, default=None,
                    help="早停: 连续多少轮 valid SI-SNRi 没有刷新最好成绩就停止, 0=不早停")
    ap.add_argument("--save-each", action="store_true", default=None,
                    help="每轮额外存一份 epoch_XXX.pt 快照(默认只保留 last.pt + best.pt 两个文件)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    P = resolve_paths(cfg)

    # ---- 参数取值: 命令行 > config > 内置默认
    data_root = Path(args.data) if args.data is not None else P.data
    manifest = data_root / P.manifest.relative_to(P.data) if data_root != P.data else P.manifest
    out_dir = Path(args.out) if args.out is not None else P.runs
    epochs = int(pick(args.epochs, cfg.train.epochs))
    batch_size = int(pick(args.batch_size, cfg.train.batch_size))
    lr = float(pick(args.lr, cfg.train.lr))
    segment = float(pick(args.segment, cfg.train.segment))
    sr = int(pick(args.sr, cfg.data.sr))
    num_workers = int(pick(args.num_workers, cfg.train.num_workers))
    device_arg = str(pick(args.device, cfg.train.device))
    resume = pick(args.resume, cfg.train.resume)
    max_steps = int(pick(args.max_steps, cfg.train.max_steps))
    patience = int(pick(args.patience, cfg.train.patience))
    save_each = bool(args.save_each or cfg.train.save_each)
    seed = int(pick(args.seed, cfg.train.seed))
    clip_grad = float(cfg.loss.clip_grad)

    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if (device_arg == "auto" and torch.cuda.is_available())
        else (torch.device(device_arg) if device_arg != "auto" else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "配置里指定了 CUDA，但当前 torch 用不了 GPU。常见原因：\n"
            "  1) 装的是 CPU 版 torch -> 重装 CUDA 版：\n"
            "     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
            "  2) 显卡驱动太旧或未安装\n"
            "想先用 CPU 跑：python train_demo.py --device cpu"
        )

    use_amp = device.type == "cuda"
    scaler = _make_scaler(device.type, use_amp)

    ds_kwargs = dict(
        segment_sec=segment,
        sample_rate=sr,
        remix=cfg.dataset.remix,
        snr_range=tuple(cfg.dataset.snr_range),
        random_gain=tuple(cfg.dataset.random_gain),
        repeat_if_short=cfg.dataset.repeat_if_short,
    )
    train_csv = manifest / f"{cfg.paths.splits[0]}{cfg.paths.manifest_suffix}"
    valid_csv = manifest / f"{cfg.paths.splits[1]}{cfg.paths.manifest_suffix}"
    for csv_path in (train_csv, valid_csv):
        if not csv_path.exists():
            raise SystemExit(f"找不到 {csv_path}，请先运行 prepare.py（或改 config 的 paths.data）")

    train_ds = VoiceBankDataset(train_csv, training=True, **ds_kwargs)
    valid_ds = VoiceBankDataset(valid_csv, training=False, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                              num_workers=num_workers, pin_memory=device.type == "cuda",
                              persistent_workers=num_workers > 0)

    model = MaskUNet(
        n_fft=int(cfg.model.n_fft),
        hop=int(cfg.model.hop),
        win=int(cfg.model.win),
        channels=tuple(cfg.model.channels),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_si_snri, start_epoch, no_improve = -1e9, 1, 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        # 跨设备续训(例如 CPU 上训的 last.pt 拿到 GPU 机器上接着训):
        # optimizer 的动量状态不会自动跟着模型搬家, 必须手动搬,
        # 否则第 1 个 step 就会报 device mismatch。
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        best_si_snri = ckpt.get("best", -1e9)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"从 {resume} 恢复, 从第 {start_epoch} 轮继续")

    params = sum(p.numel() for p in model.parameters())
    print(f"设备 {device}  参数量 {params/1e6:.2f}M  train={len(train_ds)}  valid={len(valid_ds)}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)}  显存 {prop.total_memory/1e9:.1f} GB  "
              f"混合精度(AMP): {'开' if use_amp else '关'}")
    else:
        print("  CPU 模式: AMP 已自动关闭, 建议加大 batch-size 前先确认内存够用")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        total, snr_sum, mag_sum, steps = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            noisy = batch["noisy"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(device.type, use_amp):
                est = model(noisy)
                loss, snr_l, mag_l = combined_loss(est, clean, noisy,
                                                   n_fft=model.n_fft, hop=model.hop,
                                                   w_snr=float(cfg.loss.w_snr),
                                                   w_mag=float(cfg.loss.w_mag))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()); snr_sum += float(snr_l.item()); mag_sum += float(mag_l.item()); steps += 1
            if step % 100 == 0:
                print(f"    step {step}  loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                print(f"    到达 --max-steps {max_steps}, 提前结束本轮", flush=True)
                break

        metrics = evaluate(model, valid_loader, device)
        sched.step(metrics["si_snri"])
        improved = metrics["si_snri"] > best_si_snri
        if improved:
            best_si_snri = metrics["si_snri"]
            no_improve = 0
        else:
            no_improve += 1

        # last.pt: 每轮无条件覆盖 -> 断点续训用
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best_si_snri,
                    "si_snri": metrics["si_snri"], "history_no_improve": no_improve},
                   out_dir / "last.pt")
        # best.pt: 只有刷新最好成绩时才覆盖 -> 这就是被"特殊标记"出来的最优模型
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch, "best": best_si_snri,
                        "base_si_snr": metrics["base_si_snr"], "si_snr": metrics["si_snr"]},
                       out_dir / "best.pt")
        # 可选: 每轮留一份独立快照, 方便事后挑某一轮
        if save_each:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "si_snri": metrics["si_snri"]}, out_dir / f"epoch_{epoch:03d}.pt")

        print(f"epoch {epoch:3d}/{epochs}  loss {total/max(steps,1):.4f}  "
              f"snr_loss {snr_sum/max(steps,1):.2f}dB  mag {mag_sum/max(steps,1):.4f}  "
              f"| valid SI-SNR {metrics['si_snr']:.2f} (base {metrics['base_si_snr']:.2f})  "
              f"SI-SNRi {metrics['si_snri']:.2f}{'  <- best' if improved else f'  (no improve {no_improve})'}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  | {time.time()-t0:.0f}s", flush=True)

        if patience and no_improve >= patience:
            print(f"\n连续 {no_improve} 轮没有刷新最好成绩, 触发早停(第 {epoch} 轮结束)。"
                  f"最佳 SI-SNRi = {best_si_snri:.2f} dB", flush=True)
            break

    print(f"\n完成。最佳 SI-SNRi = {best_si_snri:.2f} dB，最优权重: {out_dir / 'best.pt'}")
    print(f"评估 PESQ/STOI: python evaluate.py --ckpt {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
