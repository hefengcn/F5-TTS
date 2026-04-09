# F5-TTS 系统设计文档 (SDD)

**版本**: 1.0
**日期**: 2026-04-09
**项目**: F5-TTS — A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

---

## 1. 系统架构总览

### 1.1 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                      用户接口层 (Interface)                   │
│  Gradio Web UI · CLI · Python API · Socket Server · Triton  │
├─────────────────────────────────────────────────────────────┤
│                      推理引擎层 (Inference)                   │
│  infer_process · chunk_text · batch_process · streaming     │
├─────────────────────────────────────────────────────────────┤
│                      模型核心层 (Model Core)                  │
│  CFM (Flow Matching) · Backbone (DiT/UNetT/MMDiT) · Modules│
├─────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)              │
│  Vocoder · Tokenizer · MelSpec · ASR (Whisper) · DataLoader │
├─────────────────────────────────────────────────────────────┤
│                      训练管线层 (Training)                    │
│  Trainer (Accelerate) · EMA · Dataset · Dynamic BatchSampler│
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心数据流

```
                        推理数据流
                        ════════

  参考音频 ──→ 预处理 (去静音/截断/重采样)
  参考文本 ──→ (如空则 ASR 转录)
       │
       ├──→ Mel 频谱提取 (Vocos/BigVGAN)
       │
       ├──→ 文本 Tokenization (pinyin/char/byte/custom)
       │
       └──→ CFM.sample() ──────────────────────────────┐
            │                                            │
            │  for each ODE step:                        │
            │    t ∈ [0, 1]                              │
            │    pred = Backbone(noised_mel, cond_mel,   │
            │                    text_emb, timestep)     │
            │    + CFG: pred + (pred - null) * strength  │
            │    update noised_mel                       │
            │                                            │
            └──→ 预测的 Mel 频谱 ──→ Vocoder ──→ WAV 音频
```

### 1.3 训练数据流

```
                        训练数据流
                        ════════

  数据集 (HF/Arrow) ──→ HFDataset / CustomDataset
       │
       ├──→ 音频滤波 (0.3s ~ 30s)
       ├──→ 重采样 → 24kHz
       ├──→ Mel 频谱提取
       │
       └──→ DynamicBatchSampler (按帧数分组)
              │
              └──→ CFM.forward()
                    │
                    ├──→ 随机 span mask (frac_lengths_mask)
                    ├──→ x0 = Gaussian 噪声
                    ├──→ x1 = 真实 mel
                    ├──→ φ_t = (1-t)·x0 + t·x1
                    ├──→ flow = x1 - x0
                    ├──→ pred = Backbone(φ_t, cond, text, t)
                    ├──→ Loss = MSE(pred, flow) [仅 mask 区域]
                    │
                    └──→ backward → optimizer → EMA update
```

---

## 2. 核心模块设计

### 2.1 Conditional Flow Matching (CFM)

**文件**: `src/f5_tts/model/cfm.py`

CFM 是整个系统的生成模型核心，实现条件流匹配算法。

**类**: `CFM(nn.Module)`

**关键设计决策**:

| 设计 | 选择 | 原因 |
|------|------|------|
| 流匹配目标 | OT-CFM (Optimal Transport) | 直线路径 x0→x1，训练更稳定 |
| ODE 求解器 | Euler (默认), Midpoint | Euler 简单高效，Midpoint 更精确 |
| Classifier-Free Guidance | audio_drop_prob=0.3, cond_drop_prob=0.2 | 双重 dropout：仅丢音频 vs 同时丢音频+文本 |
| 采样策略 | Sway Sampling + EPSS | Sway 重新分配 ODE 步数；EPSS 支持极低 NFE (4步) |
| 噪声参数 sigma | 0.0 | 无额外噪声注入 |

**接口**:

```python
# 训练
def forward(self, x1, text, mel_spec=None, ...):
    """输入真实 mel + 文本，输出 loss"""

# 推理
def sample(self, cond, text, duration, ...):
    """输入条件 mel + 文本 + 时长，输出预测 mel"""
```

**随机 Span Mask 机制**:
- 训练时随机选择 70%~100% 的帧作为 mask 区域
- 只在 mask 区域计算 loss（模型学习预测被遮挡部分）
- 这使模型既能做完整生成，也能做局部编辑（语音编辑的基础）

### 2.2 Backbone 架构

系统支持三种可插拔的 Transformer backbone。

#### 2.2.1 DiT — Diffusion Transformer（主架构）

**文件**: `src/f5_tts/model/backbones/dit.py`

```
输入: [noised_mel, cond_mel, text_embedding]
       ↓ concatenation + Linear Projection
       ↓ + ConvPositionEmbedding (2×Conv1d, k=31)
       ↓
  ┌──────────────────────────┐
  │  DiTBlock × depth (22)   │
  │  ┌────────────────────┐  │
  │  │ AdaLayerNorm       │  │ ← timestep conditioning
  │  │ Multi-Head Attn    │  │ ← RoPE on all heads
  │  │ Gate Residual      │  │
  │  │ LayerNorm          │  │
  │  │ FFN                 │  │
  │  │ Gate Residual      │  │
  │  └────────────────────┘  │
  └──────────────────────────┘
       ↓
  AdaLayerNorm_Final (shift + scale)
       ↓
  Linear → 预测的 flow vector
```

**F5-TTS v1 特有增强**:
- **ConvNeXtV2 层** (4 层): 在 TextEmbedding 中加入深度可分离卷积，增强局部时序建模
- **text_mask_padding**: 对 padding 位置屏蔽，避免无效信息干扰
- **全头 RoPE**: pe_attn_head=None，所有注意力头都使用旋转位置编码
- **Zero-init**: AdaLN 调制层和输出投影初始化为零（DiT 设计原则，确保训练初始等于恒等变换）

**F5TTS_v1_Base 配置**: dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, text_dim=512, conv_layers=4

#### 2.2.2 UNetT — Flat UNet Transformer

**文件**: `src/f5_tts/model/backbones/unett.py`

E2-TTS 架构，与 DiT 的关键区别:

| 特性 | DiT | UNetT |
|------|-----|-------|
| 归一化 | AdaLayerNorm (timestep 调制) | RMSNorm (timestep 作为额外 token) |
| Skip Connection | 无 | 首尾层对称连接 (concat/add) |
| RoPE | 全头或指定头 | 仅 pe_attn_head=1 |
| Text/Audio 混合 | 在 InputEmbedding 层拼接 | 在 InputEmbedding 层混合 |
| 配置 | depth=22 | depth=24, ff_mult=4 |

#### 2.2.3 MMDiT — Multi-Modal DiT

**文件**: `src/f5_tts/model/backbones/mmdit.py`

基于 Stable Diffusion 3 的双流架构:

```
  Audio Stream ──→ AudioEmbedding ──→ ┐
                                       │ JointAttention (concat QKV)
  Text Stream  ──→ TextEmbedding  ──→ ┘
                                       │
                    ← Gate Residual ──  │
                    ← FFN (各自独立) ──  │
```

- 音频和文本各有独立的 Transformer 流
- 通过 JointAttention 交叉注意力交互
- 最后一层 text stream 跳过 FFN (context_pre_only)

### 2.3 构建模块

**文件**: `src/f5_tts/model/modules.py`

| 模块 | 用途 |
|------|------|
| `MelSpec` | Mel 频谱提取（缓存的 STFT 基矩阵 + Hann 窗） |
| `ConvNeXtV2Block` | 深度可分离 Conv7 + LN + PW + GELU + GRN + PW + 残差 |
| `ConvPositionEmbedding` | 2×Conv1d (k=31, groups=16) + Mish，替代绝对位置编码 |
| `GRN` | Global Response Normalization (ConvNeXt-V2 组件) |
| `AdaLayerNorm` | 自适应 Layer Norm，生成 6 个调制参数 (shift/scale/gate × 2) |
| `DiTBlock` | AdaLN → Attn → Gate残差 → LN → FFN → Gate残差 |
| `Attention` | 多头注意力 + 可选 QK-norm + 可选 FlashAttention |
| `FeedForward` | Linear → GELU → Dropout → Linear |

**注意力后端**: `"torch"` (PyTorch SDPA) 或 `"flash_attn"` (FlashAttention varlen)

### 2.4 Tokenizer

**文件**: `src/f5_tts/model/utils.py`

| 类型 | 机制 | 适用场景 |
|------|------|----------|
| `pinyin` | 中文→拼音+声调，英文/符号保留 | 中文/中英混合（默认） |
| `char` | 字符级编码 | 特定语言 |
| `byte` | UTF-8 字节级 (vocab_size=256) | 通用但信息密度低 |
| `custom` | 自定义 vocab.txt 文件 | 新语言适配 |

---

## 3. 推理引擎设计

### 3.1 推理管线

**文件**: `src/f5_tts/infer/utils_infer.py`

```
用户输入
  │
  ├── ref_audio, ref_text, gen_text
  │
  ↓
preprocess_ref_audio_text()
  │ ├── 去除首尾静音
  │ ├── 截断至 12 秒
  │ ├── 重采样至 24kHz
  │ └── ref_text 为空时调用 Whisper 转录
  │
  ↓
infer_process()
  │ ├── chunk_text(): 按句子边界分块，每块 ≤ max_chars
  │ │     max_chars = ref_audio_len / ref_text_len × gen_text_len
  │ ├── 对每个文本块:
  │ │     ├── 拼接 ref_text + gen_text
  │ │     ├── Tokenization → token indices
  │ │     ├── 估算时长: duration = ref_len + (ref_len/ref_text × gen_text/speed)
  │ │     ├── CFM.sample(): ODE 求解 → 预测 mel
  │ │     └── Vocoder decode: mel → waveform
  │ └── 交叉淡入淡出拼接各块
  │
  ↓
输出: wav (numpy), sr (24000), spec (mel)
```

### 3.2 流式推理

**文件**: `src/f5_tts/socket_server.py`

```
客户端 ──TCP──→ Socket Server
                  │
                  ├── 接收文本
                  ├── 自适应分块: 首块短 (快首包), 后续块正常
                  ├── 逐块推理 + 流式返回
                  │     每块: CFM.sample() → Vocoder → 2048 samples → 发送
                  └── 发送 END 标记
                  │
                  └── AudioFileWriterThread: 异步写入完整文件
```

**首包优化**: 第一块使用更短的字符数（较少文本 → 较短推理时间 → 更快首包响应）。

### 3.3 模型加载

```python
load_vocoder(mel_spec_type, is_local, local_path, device)
  → Vocos: 下载 hf://charactr/vocos-mel-24khz
  → BigVGAN: 下载 hf://nvidia/bigvgan_v2_24khz_100band_256x

load_model(model_cls, model_arc, ckpt_path, ...)
  → 构造 CFM(backbone=DiT(...), ...)
  → 加载 checkpoint (.pt 或 .safetensors)
  → 提取 EMA 权重作为推理权重
```

---

## 4. 训练系统设计

### 4.1 训练器

**文件**: `src/f5_tts/model/trainer.py`

```
Trainer
  │
  ├── Accelerate (DDP 多 GPU)
  │
  ├── DataLoader
  │     ├── batch_size_type="frame": DynamicBatchSampler
  │     │     按帧长度排序 → 分组为等帧数 batch → 打乱 batch 顺序
  │     └── batch_size_type="sample": 固定 batch_size
  │
  ├── 优化器: AdamW (fused=True)
  │     LR Schedule: Linear Warmup → Linear Decay
  │
  ├── EMA: ema_pytorch.EMA (指数移动平均)
  │     推理使用 EMA 权重，非训练权重
  │
  ├── Checkpoint 管理:
  │     ├── save_per_updates: 每 N 步保存
  │     ├── keep_last_n_checkpoints: 保留最近 N 个
  │     ├── last_per_updates: 最近 N 步的细粒度保存
  │     └── 包含: model, optimizer, EMA, scheduler, epoch/step 状态
  │
  └── 日志: WandB / TensorBoard
        ├── loss 曲线
        └── log_samples: 每个 checkpoint 自动生成样本音频
```

**默认训练超参** (F5TTS_v1_Base):
- epochs=11, lr=7.5e-5, warmup=20000 updates
- batch_size=38400 frames/GPU, max_samples=64
- grad_accumulation=1, max_grad_norm=1.0

### 4.2 数据管线

**文件**: `src/f5_tts/model/dataset.py`

```
数据源                     处理                     输出
─────                     ─────                     ────

HuggingFace Dataset ──→ HFDataset ─────────────→ collate_fn
Local Arrow/Parquet ──→ CustomDataset ─────────→ collate_fn
                            │
                            ├── 音频时长过滤 (0.3s ~ 30s)
                            ├── 重采样至 24kHz (缓存 resampler)
                            ├── Mel 频谱提取
                            └── 返回: (mel, mel_len, text, text_len)

DynamicBatchSampler:
  ├── 按帧长度排序
  ├── 按 frames_threshold 分组为 batch
  ├── 打乱 batch 顺序 (确定性种子)
  └── 确保每个 batch 的总帧数不超过阈值
```

### 4.3 配置系统

Hydra + OmegaConf YAML 配置:

```yaml
# src/f5_tts/configs/F5TTS_v1_Base.yaml
hydra:
  run:
    dir: ckpts/${model.name}_.../${now:%Y-%m-%d}/${now:%H-%M-%S}

model:
  name: F5TTS_v1_Base
  backbone: DiT          # → hydra.utils.get_class("f5_tts.model.backbones.dit.DiT")
  tokenizer: pinyin
  arch:
    dim: 1024
    depth: 22
    heads: 16
    conv_layers: 4        # ConvNeXtV2 层数
  mel_spec:
    target_sample_rate: 24000
    n_mel_channels: 100
    mel_spec_type: vocos  # vocos | bigvgan

datasets:
  batch_size_per_gpu: 38400
  batch_size_type: frame  # frame | sample
```

模型类通过 `hydra.utils.get_class` 动态实例化，实现 backbone 可插拔。

---

## 5. API 设计

### 5.1 Python API

**文件**: `src/f5_tts/api.py`

```python
class F5TTS:
    def __init__(self,
        model="F5TTS_v1_Base",       # 模型名
        ckpt_file="",                 # 自定义 checkpoint 路径
        vocab_file="",                # 自定义 vocab 路径
        ode_method="euler",           # ODE 求解器
        use_ema=True,                 # 使用 EMA 权重
        vocoder_local_path=None,      # 本地 vocoder 路径
        device=None,                  # 设备 (auto-detect)
        hf_cache_dir=None             # HF 缓存目录
    )

    def infer(self,
        ref_file,                     # 参考音频路径
        ref_text,                     # 参考文本 (空=ASR)
        gen_text,                     # 待合成文本
        nfe_step=32,                  # 去噪步数
        speed=1.0,                    # 语速
        cfg_strength=2,               # CFG 强度
        sway_sampling_coef=-1,        # Sway sampling
        cross_fade_duration=0.15,     # 交叉淡入淡出
        target_rms=0.1,               # 响度归一化
        fix_duration=None,            # 固定时长 (秒)
        remove_silence=False,         # 去静音
        seed=None                     # 随机种子
    ) -> Tuple[numpy.ndarray, int, numpy.ndarray]  # (wav, sr, mel_spec)
```

### 5.2 CLI 接口

| 命令 | 用途 |
|------|------|
| `f5-tts_infer-cli` | 推理（flags 或 TOML 配置） |
| `f5-tts_infer-gradio` | Gradio Web UI |
| `f5-tts_finetune-cli` | CLI 微调 |
| `f5-tts_finetune-gradio` | Gradio 微调 UI |

### 5.3 Socket 协议

```
客户端 → 服务端:
  文本字符串 (UTF-8 编码)

服务端 → 客户端:
  音频数据块: struct.pack(f"{len(chunk)}f", *chunk)
  结束标记: b"END"
```

---

## 6. 部署架构

### 6.1 单机部署

```
┌──────────────────────┐
│  Docker Container    │
│  ┌────────────────┐  │
│  │ Gradio Server  │  │  ← :7860
│  │   (Flask/uvicorn)│ │
│  └───────┬────────┘  │
│          │            │
│  ┌───────▼────────┐  │
│  │ F5TTS Model    │  │
│  │ (PyTorch/CUDA) │  │
│  └────────────────┘  │
│          │            │
│  ┌───────▼────────┐  │
│  │ GPU (NVIDIA)   │  │
│  └────────────────┘  │
└──────────────────────┘
```

### 6.2 生产部署 (Triton + TensorRT-LLM)

**文件**: `src/f5_tts/runtime/triton_trtllm/`

```
┌─────────────────────────────────────────────────────┐
│                  Triton Inference Server             │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ f5_tts      │  │ vocoder     │  │ (ensemble)  │ │
│  │ (Python BE) │  │ (ONNX/TRT) │  │             │ │
│  │             │  │             │  │             │ │
│  │ TextEmb(CPU)│  │             │  │             │ │
│  │    │        │  │             │  │             │ │
│  │ TRT-LLM     │  │             │  │             │ │
│  │ Engine(GPU) │──│ Mel→Wave    │  │             │ │
│  │ 32 NFE loop │  │             │  │             │ │
│  └──────┬──────┘  └─────────────┘  └─────────────┘ │
│         │                                           │
│  ┌──────▼──────┐                                   │
│  │ GPU (L20)   │                                   │
│  └─────────────┘                                   │
└──────────┬──────────────┬──────────────────────────┘
           │              │
    ┌──────▼──────┐ ┌─────▼─────┐
    │ HTTP Client │ │ gRPC      │
    │ (REST API)  │ │ Client    │
    └─────────────┘ └───────────┘
```

**TRT-LLM 加速原理**:
- DiT backbone 编译为 TensorRT 引擎（GPU 上执行）
- Text Embedding 保留在 CPU/PyTorch（灵活，支持动态 vocab）
- 32 步 ODE 循环由 Python 编排，每步调用 TRT 引擎
- 支持 Tensor Parallelism（跨 GPU 切分注意力权重）
- 支持 FP16/BF16/FP8 量化

**性能基准** (L20 GPU, 26 条测试音频, 16 NFE):

| 模式 | 并发 | 平均延迟 | RTF |
|------|------|----------|-----|
| Client-Server (TRT-LLM) | 2 | 253ms | 0.0394 |
| Offline TRT-LLM | 1 | - | 0.0402 |
| Offline PyTorch | 1 | - | 0.1467 |

---

## 7. 关键设计决策

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| 生成模型 | Flow Matching | Diffusion (DDPM/DDIM) | 直线路径 ODE，训练更高效 |
| Backbone | DiT + ConvNeXtV2 | UNet, MMDiT | ConvNeXt 增强局部建模，训练更快 |
| 位置编码 | RoPE | Sinusoidal, ALiBi | 外推性好，无固定长度限制 |
| 时序条件 | AdaLayerNorm | Cross-attention, FiLM | DiT 标准做法，效果好 |
| Vocoder | Vocos (默认) | BigVGAN, HiFi-GAN | 轻量、高质量 |
| 分词 | 拼音 (中文) | 字符, BPE | 拼音包含声调信息，更适合 TTS |
| 训练框架 | Accelerate + Hydra | PyTorch Lightning | 更灵活，社区更广泛 |
| 部署加速 | TensorRT-LLM | ONNX Runtime, torch.compile | NVIDIA GPU 最佳性能 |
| 长文本策略 | 分块 + 交叉淡入淡出 | 级联, 全量推理 | 平衡质量与显存 |
| EMA | 推理用 EMA 权重 | 直接用训练权重 | EMA 更稳定、质量更好 |

---

## 8. 文件结构

```
src/f5_tts/
├── api.py                          # Python API (F5TTS 类)
├── socket_server.py                # TCP 流式推理服务
├── socket_client.py                # TCP 流式客户端
├── configs/                        # Hydra YAML 配置
│   ├── F5TTS_v1_Base.yaml          # 主力模型配置
│   ├── F5TTS_v1_Small.yaml         # 小模型配置
│   ├── F5TTS_Base.yaml             # v0 模型配置
│   ├── F5TTS_Small.yaml
│   ├── E2TTS_Base.yaml             # E2-TTS 配置
│   └── E2TTS_Small.yaml
├── model/                          # 模型核心
│   ├── cfm.py                      # 条件流匹配
│   ├── modules.py                  # 基础构建块
│   ├── dataset.py                  # 数据集 + 采样器
│   ├── trainer.py                  # 训练器
│   ├── utils.py                    # 分词器 + 工具函数
│   └── backbones/                  # 可插拔 backbone
│       ├── dit.py                  # DiT (F5-TTS)
│       ├── unett.py                # UNetT (E2-TTS)
│       └── mmdit.py                # MMDiT (SD3-style)
├── infer/                          # 推理相关
│   ├── utils_infer.py              # 推理工具函数
│   ├── infer_cli.py                # CLI 推理入口
│   ├── infer_gradio.py             # Gradio 推理 UI
│   ├── speech_edit.py              # 语音编辑
│   ├── SHARED.md                   # 社区模型注册表
│   ├── README.md                   # 推理文档
│   └── examples/                   # TOML 示例配置
│       ├── basic/                  # 基础推理示例
│       └── multi/                  # 多说话人示例
├── train/                          # 训练相关
│   ├── train.py                    # Hydra 训练入口
│   ├── finetune_cli.py             # CLI 微调
│   ├── finetune_gradio.py          # Gradio 微调 UI
│   ├── datasets/                   # 数据准备脚本
│   └── README.md                   # 训练文档
├── eval/                           # 评估工具
│   ├── eval_infer_batch.py         # 批量推理
│   ├── eval_librispeech_test_clean.py
│   ├── eval_seedtts_testset.py
│   ├── eval_utmos.py
│   └── utils_eval.py               # WER, SIM 指标
└── runtime/                        # 生产部署
    └── triton_trtllm/              # Triton + TRT-LLM
        ├── model_repo_f5_tts/      # Triton 模型仓库
        ├── convert_checkpoint.py   # PyTorch → TRT 转换
        ├── client_grpc.py          # gRPC 客户端
        ├── client_http.py          # HTTP 客户端
        └── README.md
```
