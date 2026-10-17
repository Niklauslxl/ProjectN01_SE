# VoiceBank-DEMAND 下载与训练准备脚本

```bash
# 全新环境：装依赖
pip install -r requirements.txt

# 0) 先看一眼当前生效的路径与超参（全部来自 config/default.yaml）
python config.py


# 1) 下载（约 5.2 GB，支持断点续传）
python download.py

# 2) 解压 + 重采样到 16 kHz + 生成清单
python prepare.py

# 3) 先自检数据链路
python dataset.py

# 4) 训练基线
python train_demo.py

# 5) 评估（默认自动用 config 里 paths.runs 下的 best.pt）
python evaluate.py
```

## 配置系统（config/ + config.py）

**代码里没有硬编码路径**，全部集中在 YAML：

```
config/
  default.yaml   # 主配置：路径、下载源、数据、模型、训练、评估
  quick.yaml     # 冒烟预设（1 epoch / 10 step / 单线程，几十秒跑完）
  data48k.yaml   # 48 kHz 数据预设（与 16k 结果不可比）
  gpu.yaml       # GPU 训练预设（cuda / batch 32 / 8 workers / 输出到 runs/vbd_gpu）
```

`config.py` 负责加载与解析，提供三个能力：

| 能力 | 用法 |
|---|---|
| 切换预设 | `--config quick`、`--config data48k`、或环境变量 `VBD_CONFIG=config/data48k.yaml` |
| 临时改一项 | `--set train.epochs=5 --set paths.data=./data48k`（可写多个） |
| 查看生效值 | `python config.py [--config xxx] [--set ...]` |

**优先级：命令行参数 > `--set` > YAML > `config.py` 内置默认值。**

预设文件只需写与 `default.yaml` 不同的项，其余自动继承（递归深合并）。

### config/default.yaml 里 paths 段的含义

```yaml
paths:
  root: "."             # 所有相对路径的基准；改成 "D:/data" 就能整体搬走
  downloads: "downloads"
  data: "data_real"     # wav 与 manifest 都在这下面
  runs: "runs/vbd_base" # checkpoint 输出目录
  enhanced: "enhanced"  # evaluate 导出音频目录
  wav_dir: "wav{sr}k"   # 目录命名规则，{sr}=kHz 数、{spk}=说话人数
  manifest_dir: "manifest"
  zip: { ... }          # download.py 下载出的 zip 文件名
  splits: [train, valid, test]
  kinds:  [clean, noisy]
```

！！！改 `paths.root` 一处就能把整个项目的数据搬到别的盘；改 `splits`/`kinds` 能整体改目录结构。！！！

## 数据说明

| 子集 | 说话人 | 条数 | 时长 | 采样率 | SNR |
|---|---|---|---|---|---|
| 28spk train | 14 女 + 14 男（英式） | 11572 | ~9.4 h | 48 kHz | 0/5/10/15 dB |
| 28spk test | 2 人 | 824 | ~0.6 h | 48 kHz | 2.5/7.5/12.5/17.5 dB |
| 56spk train | 28 女 + 28 男（含苏格兰、美式） | 28409 | ~23 h | 48 kHz | 同上 |

- 干净语音取自 **VCTK**，噪声来自 **DEMAND**（8 种真实环境噪声）+ 2 种人工噪声（speech-shaped、babble）。
- 训练集噪声只有 10 种、4 档 SNR，**很容易让模型记住噪声**，这是这个数据集最大的局限。本项目的
  `dataset.py` 里默认开启 **SNR 重混合(remix)** 来缓解，实测 SI-SNRi 能提升约 1 dB。
- 原始 48 kHz，绝大多数模型跑 16 kHz，`prepare.py` 默认降到 16k；要复现某些 Speech Brain 配方请用 `--sr 48000`。

## 脚本说明

### `download.py`

```bash
python download.py                                  # 目录/代理/重试次数都读 config
python download.py --spk 56                         # 56 说话人版本
python download.py --only-test                      # 只要测试集（~310 MB）
python download.py --set download.proxy=http://127.0.0.1:7890
VBD_INSECURE=1 python download.py                   # 证书校验失败时的兜底
```

DataShare 的 bitstream id 也在 `config/default.yaml` 的 `download.files` 里，官网换链只改 YAML。

每个文件下载失败会自动重试 5 次，中断后重跑会走 HTTP Range 续传，不需要重下。 -->

### `prepare.py`

```bash
python prepare.py                                   # 全部读 config
python prepare.py --config data48k                  # 48 kHz 一套独立目录
python prepare.py --val-ratio 0.1                   # 按说话人随机划分
python prepare.py --set data.workers=16             # 加速解压
```

产出：

```
data/
  wav16k/           # 重采样后的音频（不传则用 wav48k/）
    train/{clean,noisy}/p226_001.wav ...
    valid/{clean,noisy}/
    test/{clean,noisy}/
  manifest/
    train.csv  valid.csv  test.csv   # utt_id, clean_path, noisy_path, n_samples, duration
    summary.json
```

### `dataset.py`

- 固定长度随机裁剪 + 短句重复拼接（训练时），测试时从开头裁剪，方便组 batch。
- `remix_snr()`：把 `noisy - clean` 的噪声分量重新加权到随机 SNR（默认 0~20 dB），
  等价于瞬间把 4 档 SNR 扩成连续分布。不想用就传 `remix=False`。

### `train_demo.py`

频域复数掩码 U-Net（约 1.5M 参数，512 点 FFT / 128 hop），损失为
`0.8 × SI-SNR + 0.2 × 幅度 L1`，Adam + ReduceLROnPlateau，自动混合精度（仅 CUDA）。

```bash
python train_demo.py                                          # 全部读 config
python train_demo.py --epochs 40 --batch-size 32 --lr 5e-4
python train_demo.py --set train.patience=8                   # 早停
python train_demo.py --set train.save_each=true               # 每轮留快照
python train_demo.py --resume runs/vbd_base/last.pt
python train_demo.py --config quick                           # 冒烟：1 epoch / 10 step
```

日志每轮打印验证集 SI-SNR 和 SI-SNRi，`<- best` 表示本轮刷新了最好成绩。

#### 早停机制

**默认没有早停**，`--epochs 40` 就会老老实实跑满 40 轮。只有
ReduceLROnPlateau（patience=3，学习率 ×0.5）在起作用，它只降学习率、不停训练。

需要早停就加 `--patience N`：连续 N 轮 `valid SI-SNRi` 没有刷新最好成绩就收工。
CPU 上建议 `--epochs 40 --patience 8`，日志会打印 `(no improve N)` 方便观察还剩几次机会。

#### 每个 epoch 的模型怎么存

| 文件 | 写入时机 | 用途 |
|---|---|---|
| `last.pt` | **每轮无条件覆盖** | 断点续训（`--resume`），含优化器状态与 `history_no_improve` |
| `best.pt` | 仅当 valid SI-SNRi 刷新最好成绩时覆盖 | **就是被标记出来的最优模型**，评估一律用它 |
| `epoch_XXX.pt` | 只有加 `--save-each` 才生成 | 事后挑某一轮的权重 |

即：**默认不会存 40 份模型**，全程只有 `last.pt` 和 `best.pt` 两个文件（各约 6 MB）。
PyTorch 里没有 `.ckpt` 版本的 EarlyStopping 回调（那是 Lightning/Keras 的东西），
本项目是自己数 `no_improve` 计数器实现的。

### `evaluate.py`

测试集整句推理，输出 SI-SNR / PESQ(WB) / STOI。`pesq`、`pystoi` 装不上时自动跳过，SI-SNR 始终可用。

```bash
python evaluate.py --ckpt runs/vbd_base/best.pt --data ./data --dump ./enhanced
```

`--dump` 会把 clean / noisy / est 三份音频都导出来，方便人工听。

注意 PESQ / STOI 默认只对**前 20 条**计算（PESQ 很慢），SI-SNR 则对全部条目计算，
所以 `--limit 20` 的结果和 `--limit 0`（全量）的 SI-SNR 会有差异，对比时要保持同一口径。

## 网页版演示（录音 / 上传 → 语谱图 + 试听）

一个完全离线的本地网页，`serve.py` + `web/`，只监听 127.0.0.1，不请求任何外部资源。

```bash
conda activate py39
python serve.py                      # 然后浏览器打开 http://127.0.0.1:8000
python serve.py --port 8080 --ckpt runs/real_base/best.pt
```


-----------------------------------------------------------------------------------------------------------------------
功能：

| 区域 | 内容 |
|---|---|
| 输入 | **实时录音**（麦克风，波形动画 + 计时）／**上传文件**（拖拽，wav/mp3/flac/m4a/ogg） |
| 输出 | 增强前后**语谱图**对比、原始/增强**双轨试听**、输入波形、下载增强后 WAV |
| 指标 | 噪声衰减 dB、语音保真 dB、处理耗时、实时率 RTF、音频时长 |

两个无参考指标的含义（实时录音没有干净参考，算不了 SI-SNR/STOI/PESQ）：

- **噪声衰减**：输入能量最低 10% 的帧（静音/纯噪声段）能量下降多少 dB，越大越好。
- **语音保真**：能量最高 50% 的帧（语音段）能量变化，**接近 0 最好**；明显为负
  （比如 < -3 dB）说明模型把人声也削掉了，即过度抑制。

实现要点：

- 后端零新依赖，只用标准库 `http.server` + 已有的 torch；长音频按 4 秒块、
  50% 重叠、汉宁窗加权叠加推理，避免整句峰值内存过高。
- 语谱图在后端算成 128×N 的 dB 矩阵，前端 canvas 按 viridis 风格配色绘制。
- 录音时**显式关闭浏览器的 noiseSuppression**，否则演示的是浏览器降噪而不是你的模型。
- 单次处理上限 60 秒（`serve.py` 里的 `MAX_SECONDS`）。

> 若页面打不开：本机代理软件（Clash 等）可能会拦截 127.0.0.1，
> 在代理设置里把 localhost / 127.0.0.1 加入绕过列表即可。

### 终端没有日志 / 网页提示 JSON 解析失败

这两件事是**同一个原因**：请求根本没到你正在看的那个进程。

**最常见的元凶：上次的 serve.py 没关掉，两个进程抢同一个端口。**

Windows 上两个进程可以同时监听同一个端口（后启动的那个不报错），而浏览器
可能把请求发给**上一个旧进程**。那个旧进程运行的是老代码 —— 于是你看到
"终端一条日志都没有 + 网页报各种看不懂的错"，还以为代码坏了。

**先查这个**（在另一个终端里执行）：

```bash
netstat -ano | findstr :8000
```

正常只应该有**一行** `LISTENING`。如果出现两行不同 PID 的 `LISTENING`，
就是这个问题：

```bash
taskkill /F /PID <多余的PID>
```

现在 `serve.py` 启动时会自动做这项检查，端口被占用会直接拒绝启动并告诉
你该怎么处理，不会再让你对着一个"看起来正常"的服务干瞪眼。启动成功的
输出里也会打印本次进程 PID，方便对照。

其他可能：

1. **终端提示符是否回来了。** 如果 `python serve.py` 的窗口又出现了
   `(py39_1) C:\...>`，说明服务已退出，浏览器自然连不上。
2. **地址和端口是否一致。** 服务打印 `8000`，浏览器就必须是 `8000`。
3. **换一下地址写法。** `127.0.0.1:8000` 和 `localhost:8000` 在部分机器上
   解析不同，两个都试；用 `http://` 而不是 `https://`。
4. **代理软件**（Clash / V2Ray 等）把 127.0.0.1 走了代理，请求被丢掉。
   加绕过规则或临时退出代理。
5. **不要用 `file://` 直接打开 `web/index.html`。** 页面的 js 是向
   `/api/enhance` 发请求的，必须由 `serve.py` 提供页面。
6. **换个浏览器或开无痕窗口**，某些插件会拦本地请求。

服务正常工作时，每点一次"开始增强"，终端应该出现：

```
[10:57:31] POST /api/enhance  Content-Length=131174 (128.1 KB)  来自 127.0.0.1
    Content-Type: application/json  Expect: None
    实际读到 131174 字节
    JSON 解析成功，audio 字段 131062 字符
    收到音频 93.8 KB
    解码完成 3.00s @ 16000 Hz，开始推理…
    推理完成 43 ms（RTF 0.014），正在生成语谱图…
    语谱图 128x360，返回 125.1 KB base64
    完成，总计 0.05s，已返回结果
```

如果**有** `POST /api/enhance` 这行但后面报错，看"实际读到多少字节"：
明显少于 `Content-Length` 就是请求体在传输中被截断了，请把日志贴出来。

### 网页报 `No number after minus sign in JSON`

**这不是响应被截断，是后端吐出了非法 JSON。** 报错里的 position（比如
`383935`）会落在响应中段，容易误以为是数据丢了一半。

真正的原因：录音开头/结尾的**纯静音段**帧能量为 0，`10*log10(0/0)` 得到
`-inf`，而 Python 的 `json.dumps` 会把它原样写成 `-Infinity`。这两个词
**不是合法的 JSON**，浏览器 `JSON.parse` 读到 `-` 后面跟着 `I`（不是数字），
就抛出 `No number after minus sign`。

所以这个错误**只在实时录音时出现**（录音头尾天然带静音），上传干净音频
文件时几乎不会遇到。

已从三处封死：

1. `compute_metrics` 用 `_safe_ratio_db()` 把结果夹在 [-60, 60] dB 内，
   分母为 0 时直接返回 -60，不再产生 inf/nan。
2. `_send_json` 用 `json.dumps(obj, allow_nan=False)` —— 万一将来别处再
   冒出非有限值，服务端会**当场报错**，而不是发出一个前端无法解析的响应。
3. 前端先 `resp.text()` 再手动 `JSON.parse`，并识别 `Infinity`/`NaN`
   给出中文提示；数值字段统一走 `Number.isFinite` 兜底，不会再因某一项
   异常而整页失败。录音几乎是静音时还会提前提示重录。

已实测 7 个边界用例（纯静音、极安静、DC 偏移、满幅方波、头尾长静音、
0.06 秒极短音频、正常语音）全部返回合法 JSON。

### 不戴耳机时听起来像"二次降噪"

即使用 `echoCancellation: false`，外放场景下麦克风仍会把扬声器里的**增强后
语音**一起录进去。后端对这个已经被增强过一遍的信号再增强一次，人声会被削两遍，
听感变成发闷、金属味。

两个办法：

1. **戴耳机**（最省事，演示效果也最真实）。
2. 打开 `web/app.js` 里的回声抵消：

```js
const FEEDBACK = {
  enabled: true,
  delayMs: 110,   // 用 delay.py 实测，别猜
  gain: 0.9,
};
```

先用 `delay.py` 测出扬声器 → 麦克风的真实延迟：

```bash
# 对着内置麦克风说一段话，同时用扬声器放一段音频；把两条 wav 存下来
python delay.py --ref 播放的音频.wav --rec 麦克风录到的音频.wav
```

它会用 GCC-PHAT 输出延迟毫秒数，直接填进 `delayMs`。原理是播放时发出
`g * (y(t) - y(t-delay))`，延迟处的回声被抵消，而人声几乎不受影响。

> `delay.py` 可以自检算法本身（与业务数据无关）：
> `python delay.py --self-test`，会用合成信号验证三个已知延迟，误差应为 0.00 ms。
> 如果 `gain` 调得太大而 `delayMs` 估错，会引入梳状滤波的"机器人音"，
> 宁可先用 0.3~0.5。

## 怎么测一个训好的 checkpoint

以 `runs/vbd_base/best.pt` 为例，在 conda py39 下：

```bash
conda activate py39
cd 根目录

# 1) 全量客观指标（824 条，约 45 秒；ckpt 默认就是 config 里 paths.runs 下的 best.pt）
python evaluate.py

# 2) 导出 20 条音频人工试听（noisy / est / clean 三组同名文件，直接对比听）
python evaluate.py --limit 20 --dump ./enhanced_vbd
```

导出后目录结构：

```
enhanced_vbd/
  clean/p232_001.wav   参考干净语音
  noisy/p232_001.wav   带噪输入
  est/p232_001.wav     模型输出   <- 重点听这个
```

**听什么**：噪声是否消干净、人声有没有"水下音"/金属声（过度抑制的典型症状）、
句尾静音段是否被削没了。指标好但听着发闷，往往就是过度抑制。

要换模型只需换 `--ckpt`，例如对比 1-epoch 的旧模型：

```bash
python evaluate.py --ckpt runs/real_base/best.pt --data ./data_real
```

> 别拿 `last.pt` 去测：`vbd_base/last.pt` 是第 40 轮，`best.pt` 是第 39 轮（历史最优），
> 测 last 会略差一点。`last.pt` 的唯一用途是 `--resume` 续训。

## 在 conda py39（Python 3.9）中运行

已在 **Python 3.9.25 + torch 2.8.0+cpu** 的 conda 环境实测通过，全部脚本可直接运行，无需改代码。

```bash
C:/APP/anaconda3/envs/py39/python.exe dataset.py   --data ./data_real
C:/APP/anaconda3/envs/py39/python.exe train_demo.py --data ./data_real --epochs 40 --patience 8
C:/APP/anaconda3/envs/py39/python.exe evaluate.py  --ckpt runs/vbd_base/best.pt --data ./data_real
```

> 注意：`--out` 默认是 `runs/vbd_base`。之前验证时用的是 `--out runs/real_base`，
> 所以直接 `--data ./data_real --epochs 40` 会在 `runs/vbd_base` 下新建一份结果与 old checkpoint 互不干扰。
> 想从之前那次继续训练：`--resume runs/real_base/last.pt --out runs/real_base`。

补依赖时注意一点，用 `--no-deps` 装 torchaudio，避免 pip 把你已有的 torch / numpy 版本拽乱：

```bash
C:/APP/anaconda3/envs/py39/python.exe -m pip install --no-deps torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu
C:/APP/anaconda3/envs/py39/python.exe -m pip install pystoi
```

py39 环境的已知限制：

- **`pesq` 装不上**：它是 Cython 源码包，编译需要 **Microsoft Visual C++ 14.0+**（MSVC 生成工具）。
  本机没装 → 编译失败；纯 Python 的替代品 `pypesq` 由于包太老，在新版 setuptools 下也装不上。
  实测该环境下 **STOI 可用**（`pystoi` 能装），SI-SNR 始终可用。
- 想要 PESQ 就去装 [MSVC 生成工具](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  （勾选"C++ 生成工具"），装完重启终端再 `pip install pesq`。
- `requirements.txt` 里写的 numpy/scipy 版本要求偏高（>=1.24 / >=1.10），py39 用当前已装的
  numpy 1.21.5 / scipy 1.8.0 也能正常跑——**所以不要为了"满足要求"去升级**，理由见下一节。

## 关于评估指标的一句话提醒

本机实测（1 个 epoch 的模型）出现：**SI-SNR 大涨到 35.5 dB，而 STOI 从 0.947 掉到 0.941**。

这是典型的**过度抑制**：模型把噪声连同部分语音一起抹掉，残差变小让 SI-SNR 虚高，
但可懂度反而略降。**永远别只看 SI-SNR**，务必搭配 STOI / PESQ 一起判断。

补充：把同一个模型训满 40 epoch 后（见上一节表格），SI-SNR 只再涨 1 dB，
但 **STOI 从"下降"翻成"上升"（0.947 → 0.955）** —— 说明早期的过度抑制主要是训练不足造成的。

## 依赖清单（requirements.txt）

共 **7 个第三方包**，其余全是标准库（`http.server` / `csv` / `zipfile` / `argparse` / `json`
/ `urllib` / `ssl` / `base64` / `concurrent.futures` …）。逐个对照代码里的 import：

| 包 | 必需性 | 谁在用 | 用途 |
|---|---|---|---|
| `PyYAML` | 必需 | config.py | 解析 `config/*.yaml` |
| `numpy` | 必需 | dataset / prepare / serve | 数值计算 |
| `scipy` | 必需 | prepare.py | 重采样的**回退**实现 `scipy.signal.resample_poly` |
| `soundfile` | 必需 | prepare / dataset / evaluate / serve | wav/flac 读写（底层 libsndfile，wheel 自带） |
| `torch` | 必需 | train / dataset / evaluate / serve | 模型、训练、推理 |
| `torchaudio` | 必需（但可降级） | prepare.py | 重采样的**首选**后端；导入失败会自动回退 scipy |
| `pystoi` | 可选 | evaluate.py | STOI 可懂度，纯 Python 可装 |
| `pesq` | 可选（**文件里已注释**） | evaluate.py | PESQ，Windows 需 MSVC 编译，默认不装以免拖垮整次安装 |

看代码结构就能确认这个"可降级"设计：`prepare.py:44-55` 先 try `torchaudio`，失败才走
`scipy.signal.resample_poly`；`evaluate.py:30-49` 把 `pesq` / `pypesq` / `pystoi` 全部
包在 `try/except` 里，所以**缺任何一个可选包都不会中断训练**。


### 千万不要这样装

本机 py39 环境的实测（`pip install -r requirements.txt --dry-run`）：

```
Would install numpy-2.0.2  scipy-1.13.1  pesq-0.0.4
```

即它会**把 numpy 1.21.5 升到 2.0.2、scipy 1.8.0 升到 1.13.1**。而同一个环境里还装着
tensorflow 2.8 / keras 2.8 / librosa 0.9.1 / numba 0.55，这些对 numpy 2.x 全不兼容，
升级后会连带崩掉，回退还很麻烦。

**结论：**

| 你的情况 | 该怎么做 |
|---|---|
| 只想跑本项目，且环境已能 import torch / soundfile / yaml | **什么都不用装**。缺 STOI 就单独 `pip install pystoi` |
| 想干净复现一套一模一样的环境 | 建新环境再装：`conda create -n vbd python=3.9 -y` → `conda activate vbd` → `pip install -r requirements.txt` |
| 想锁定当前能跑的精确版本 | `pip freeze > requirements-lock.txt`，以后按 lock 文件装 |

`requirements.txt` 是**"新环境从零安装"**用的，不是**"给现有环境打补丁"**用的。

### 关于 pesq 编译失败（Windows 常见）

如果安装时看到这个报错：

```
error: Microsoft Visual C++ 14.0 or greater is required.
Building wheel for pesq (pyproject.toml) ... error
ERROR: Failed building wheel for pesq
```

**含义**：`pesq` 是 Cython 源码包（内含 `dsp.c` / `pesqmod.c`），pip 需要在本机编译 C 扩展，
而 Windows 上编译必须有 MSVC 生成工具。

**真实影响**：pip 安装是「全有或全无」——只要有一个包编译失败，**整条命令会回滚**，
`torch` / `soundfile` 这些也一起装不上，白等一次（下完 240 MB 的 torch 才失败，很气）。

**本仓库已规避**：`requirements.txt` 里的 `pesq` **默认被注释掉**了，所以直接跑
`pip install -r requirements.txt` 不会再被它拖垮。

**想要 PESQ 的两条路**：

```bash
# A. 装 MSVC 生成工具（安装时勾选"使用 C++ 的桌面开发"），重启终端后：
pip install pesq

# B. 不要 PESQ。evaluate.py 会自动跳过，用 SI-SNR + STOI，评估结论不受影响
```

> 顺带提醒：`pesq` 只影响 `evaluate.py` 的一个指标，**训练、网页演示、其它全部功能都不依赖它**。

## 在 GPU 机器上训练（CUDA）

**代码只有一份，CPU 和 GPU 走的是同一套**。设备由 `train.device` 决定，默认 `"auto"`：

| 情况 | 行为 |
|---|---|
| `auto` + 检测到 CUDA | 用 GPU，自动开启 AMP 混合精度，`DataLoader` 自动开 `pin_memory` |
| `auto` + 没有 CUDA | 退回 CPU，自动关闭 AMP（GradScaler 在 CPU 上无意义） |
| 显式指定 | `--device cuda` / `--device cuda:1` / `--device cpu` |

所以搬到有显卡的机器上，**不需要改任何代码**，只要装对 torch 版本：

```bash
# 1) 必须是 CUDA 版 torch（按显卡驱动选 cu118 / cu121 / cu124）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) 确认真的能用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) 用 GPU 预设开跑（输出到 runs/vbd_gpu，不覆盖 CPU 上那份）
python train_demo.py --config gpu
```

`config/gpu.yaml` 相对 default 改了这些：

| 项 | CPU 默认 | GPU 预设 | 原因 |
|---|---|---|---|
| `train.device` | auto | cuda | 显式指定，装错 torch 时能立刻发现 |
| `train.batch_size` | 16 | 32 | GPU 吃得下，批越大 BN 统计越稳；显存紧张改 16 或 8 |
| `train.num_workers` | 4 | 8 | GPU 太快，数据加载容易成为瓶颈 |
| `train.patience` | 0 | 8 | GPU 上跑得快，早停省时间 |
| `paths.runs` | runs/vbd_base | runs/vbd_gpu | 与 CPU 结果分开存，方便对比 |

### 跨设备续训（CPU ↔ GPU）

`last.pt` / `best.pt` 里存的是**权重状态字典，不含设备信息**，`torch.load` 时统一
`map_location="cpu"`，因此：

- CPU 上训到一半的 `last.pt`，可以直接拷到 GPU 机器上 `--resume` 接着训；
- GPU 训出来的权重，也能在只有 CPU 的机器上 `evaluate.py` 评估或跑 `serve.py`。

唯一的坑已经修掉了：`opt.load_state_dict()` 不会把 Adam 的动量状态搬到新设备，
直接用会报 device mismatch。现在脚本在 resume 后会显式把 optimizer 状态 `.to(device)`。

### 速度参考

| 设备 | 单 epoch | 40 epoch |
|---|---|---|
| 本机 CPU（8 线程） | 约 18.5 分钟 | 约 12.3 小时 |
| 单张 3060（估） | 约 1~2 分钟 | 约 40~80 分钟 |

评估同理：`evaluate.py --config gpu` 跑完 824 条从 CPU 的 45 秒降到几秒。

> 如果装的是 CPU 版 torch，`--config gpu` 会在启动阶段就报错退出，并提示重装 CUDA 版
> torch——这正是预设里显式写 `cuda` 而不是 `auto` 的目的：
> **宁可立刻报错，也不要默默用 CPU 跑一晚上**。想临时用 CPU 就加 `--device cpu`。

## 常见问题

**下载太慢或断连**：直连 Edinburgh DataShare 在国内不太稳，建议配代理 `--proxy`；
断连后直接重跑同一条命令即可续传。

**PESQ 装不上**：`pip install pesq` 在 Windows 需要 Visual Studio 生成工具。
替代方案：`pip install pypesq` 或用 SpeechBrain 提供的 `AduioX`；也可以只用 STOI + SI-SNR。

**48k 还是 16k**：对比 papers 时务必看清楚，VoiceBank 的原版测试集是 48 kHz，
降采样到 16k 时结果与 48k 不可直接比较。

**Windows 下 DataLoader 报错**：`num_workers>0` 必须放在 `if __name__ == "__main__":` 保护里，
本项目的 `__main__` 已经做了；若仍有问题，先 `--num-workers 0` 排查。

**想换模型**：把 `train_demo.py` 里的 `MaskUNet` 换成自己的网络即可，
输入输出都是 `[B, T]` 的 16 kHz 波形，`dataset.py` 不需要动。

## 已验证 / 未验证

本机已用**完整真实数据**跑通全流程（Windows / Python 3.13 / torch CPU版）。

### 数据准备结果

`prepare.py --downloads ./downloads --data ./data_real --sr 16000` 实测（26 分钟处理 23144 个 wav）：

| 子集 | 条数 | 时长 | 说话人 |
|---|---|---|---|
| train | 10842 | 8.81 h | 26 |
| valid | 730 | 0.58 h | 2（p282、p287） |
| test | 824 | 0.58 h | 2（p232、p257） |

- 10842 + 730 = 11572，与官方训练集条数完全一致，无丢数据
- **train / valid / test 三组说话人零重叠**，不存在说话人泄漏
- 抽样 400 条：采样率全部 16 kHz，clean 与 noisy 长度 100% 一致
- 官方 paper 常引用 9.4 h，实测精确求和是 8.81 h，属统计口径差异

### 本机实测训练结果（CPU）

完整测试集 824 条实测，`--batch-size 16 --segment 3.0`：

| 模型 | 训练量 | valid SI-SNRi | 测试 SI-SNR | 测试 STOI |
|---|---|---|---|---|
| `runs/real_base/best.pt` | 1 epoch | 13.97 | 16.89 → 35.53 dB | 0.947 → **0.941（降）** |
| `runs/vbd_base/best.pt` | 40 epoch（best 落在第 39 轮） | 19.76 | 16.89 → 36.50 dB | 0.947 → **0.955（升）** |

`vbd_base` 那次跑满 40 轮用了约 **12.3 小时**（18.5 分钟/epoch，CPU）。

**关键观察**：SI-SNR 两个模型只差 1 dB，但 STOI 方向完全相反——
**只训 1 个 epoch 时 STOI 是下降的（过度抑制）**，训满 40 轮后 STOI 才转为上升。
所以"SI-SNR 涨那么猛是不是过拟合"这个问题，答案反而是**训练不足**。
无论跑几轮，评估都务必 SI-SNR + STOI + PESQ 一起看。

参考预期：这个小模型（0.49M）正常该训练 **30~40 个 epoch**。
本机 CPU 约 **18~27 分钟/epoch**，完整跑完要 8~18 小时；有 GPU 的话会快一个数量级。

### 压缩包校验

下载后建议先看一眼再解压，两个训练集 zip 应分别为 2371 MB / 2699 MB，
内含 **11573 个条目**（11572 条音频 + 1 个根目录），与本仓库核对逻辑一致。

## 下载速度实测（重要）

本机直连 Edinburgh DataShare 的实测速度：**42~48 KB/s，且全天稳定**。

| 目标 | 大小 | 实测耗时 |
|---|---|---|
| 仅测试集 `--only-test` | 310 MB | 约 1 小时 45 分 |
| 28 说话人完整数据 | 5.2 GB | 约 30 小时 |

所以：**有代理请务必用 `--proxy`**（通常能快几十倍），没有就挂着让它慢慢跑，
脚本随时中断、随时重跑都会续传。训练集建议睡前挂一夜。

## 真实数据验证结果

用真实测试集跑完 `prepare.py --sr 16000` 后的实测（57 秒完成 1648 个 wav）：

- 测试集 824 条、0.58 小时 —— 与官方公布的 0.6 h 吻合
- 说话人为 **p232 / p257**，即官方测试集的 1 女 1 男
- 采样率全部 16000 Hz，clean/noisy **长度 100% 一致**（配对无错位）
- 无静音、无读坏的文件
- 活跃帧 SNR 分布在 **2.25 ~ 21 dB**，覆盖官方标称的 2.5 / 7.5 / 12.5 / 17.5 dB 四档

> 注：整句 RMS 算出来的 SNR 会偏低（最低约 -3 dB），因为静音段也被算进去了。
> 这是算法口径差异，不是数据问题，用语音活跃帧计算即可回到标称区间。

## 只下载了测试集也能用

`prepare.py` 现在会自动跳过没下载的 zip。只有测试集时可以正常生成 `test.csv`
用于跑 `evaluate.py` 评估已有模型；要训练则必须补下训练集。

## 后续可以做的增强

- 用 **DEMAND / MUSAN / WHAM!** 噪声对 VCTK 重新混合，摆脱固定的 10 种噪声。
- 加 **SLR26 / SLR28** 的 RIR 做卷积，让模型对混响也鲁棒。
- 损失里再叠加 **PESQ 代理损失**或 discriminator，主观质量提升更明显（参考 MetricGAN）。
