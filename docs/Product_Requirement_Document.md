# F5-TTS 产品需求文档 (PRD)

**版本**: 1.0
**日期**: 2026-04-09
**项目**: F5-TTS — A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

---

## 1. 产品概述

### 1.1 产品定位

F5-TTS 是一个基于 Flow Matching + Diffusion Transformer 的开源文本转语音 (TTS) 系统。其核心能力是**零样本语音克隆**：仅需一段参考音频（无需微调），即可合成该说话人的语音。同时支持微调训练以适配特定说话人或语言。

### 1.2 产品愿景

提供一个**高质量、低门槛、多语言**的 TTS 解决方案，使研究人员、开发者和内容创作者能够：

- 零样本克隆任意说话人的声音
- 用极少量数据微调专属语音模型
- 在多种语言和场景中部署生产级语音合成服务

### 1.3 核心价值主张

| 价值 | 说明 |
|------|------|
| **零样本克隆** | 提供一段参考音频即可复刻声音，无需训练 |
| **高保真度** | 24kHz 采样率，100 通道 mel 频谱，接近自然人声 |
| **多语言** | 原生支持中英文，社区已扩展至 12+ 种语言 |
| **多接口** | CLI、Web UI、Python API、Socket 流式、HTTP 微服务 |
| **可微调** | Gradio/CLI 全流程微调，从数据准备到模型部署 |
| **生产就绪** | Triton + TensorRT-LLM 加速，RTF 0.04（L20 GPU） |

---

## 2. 目标用户与使用场景

### 2.1 用户画像

| 用户类型 | 需求 | 使用方式 |
|----------|------|----------|
| **AI 研究人员** | 复现论文、改进模型架构 | Python API + 训练配置 |
| **应用开发者** | 将 TTS 集成到产品中 | Python API / Socket Server / Triton 部署 |
| **内容创作者** | 生成配音、有声读物、播客 | Gradio Web UI / CLI |
| **语言社区贡献者** | 训练新语言模型 | 微调 Gradio + 数据准备工具 |
| **运维工程师** | 大规模部署 TTS 服务 | Docker / Triton / TensorRT-LLM |

### 2.2 核心使用场景

**场景 1: 快速语音合成**
> 内容创作者上传一段 5 秒的参考音频，输入待合成文本，30 秒内获得该说话人的语音输出。

**场景 2: 多角色有声书生成**
> 有声书制作者定义多个说话人（旁白、角色 A、角色 B），在文本中用标签切换角色，一键生成多人对话的有声内容。

**场景 3: 语音对话系统**
> 开发者搭建语音聊天机器人：用户说话 → ASR → LLM → TTS → 语音回复，端到端完成语音对话。

**场景 4: 新语言适配**
> 语言社区贡献者收集目标语言的语音数据，使用微调 Gradio 界面完成数据准备、训练、测试，产出新语言模型并发布到 HuggingFace。

**场景 5: 生产部署**
> 运维团队将模型部署为 HTTP 微服务，通过 Triton + TensorRT-LLM 加速，支持并发推理和动态批处理。

---

## 3. 功能需求

### 3.1 推理功能 (P0 — 核心功能)

#### FR-001: 基础 TTS 合成

- **输入**: 参考音频文件 + 参考文本 + 待合成文本
- **输出**: 合成的 WAV 音频文件（24kHz）
- **参数控制**:
  - 语速调节（0.3x ~ 2.0x）
  - NFE 步数调节（4 ~ 64，平衡质量与速度）
  - CFG 强度调节（classifier-free guidance）
  - Sway Sampling 系数
  - 交叉淡入淡出时长
  - 静音去除开关
  - 种子控制（可复现）
- **接口**: CLI (`f5-tts_infer-cli`)、Gradio UI、Python API、Socket Server
- **长文本处理**: 自动分句、分块推理、交叉淡入淡出拼接

#### FR-002: 自动语音识别 (ASR)

- 当用户不提供参考文本时，自动调用 Whisper large-v3-turbo 转录参考音频
- 支持中英文自动识别
- 结果缓存（相同音频不重复转录）

#### FR-003: 多说话人/多风格生成

- 定义多个说话人，每个说话人有独立的参考音频和参数（语速、种子）
- 在文本中用 `{StyleName}` 或 `{"name":"...", "seed":N, "speed":F}` 标签切换说话人
- 支持最多 100 个说话人定义
- 每段语音附带元数据（seed、speed），便于复现和 "cherry-pick"

#### FR-004: 语音聊天

- 集成 LLM（默认 Qwen2.5-3B-Instruct）实现对话式语音交互
- 支持麦克风输入（语音→文字→LLM→TTS→语音）和文本输入
- 可配置系统提示词（定制 AI 人格）
- 对话历史保持上下文连贯

#### FR-005: 语音编辑

- 对已有音频的指定片段进行内容替换（如替换某个词/句）
- 保持非编辑区域的原始音频不变
- 保持说话人声音一致性
- 基于 ctc-forced-aligner 计算精确的时间对齐

#### FR-006: 流式推理

- TCP Socket 服务器实时流式传输音频
- 首包优化：第一块文本使用更短的分块，加快首字响应
- 异步文件写入
- 客户端实时播放

### 3.2 训练与微调功能 (P0 — 核心功能)

#### FR-010: 数据准备

- 上传原始音频（wav/ogg/opus/mp3/flac），自动按静音切片和转录
- 或提供预组织的 `wavs/` + `metadata.csv` 目录
- 自动检查词汇覆盖率，缺失字符时扩展模型嵌入层
- 音频时长验证（1~30 秒），格式转换（统一为训练格式）

#### FR-011: 模型训练

- 支持三种模型架构: F5TTS_v1_Base / F5TTS_Base / E2TTS_Base
- 支持从头训练或基于预训练权重微调
- Hydra YAML 配置系统 + Gradio 可视化配置
- 自动批次大小估算（根据 GPU 显存和数据量）
- 训练过程可视化（WandB / TensorBoard）
- 训练中自动生成样本音频用于质量监控
- 梯度累积、混合精度（FP16/BF16）、8-bit 优化器

#### FR-012: 模型测试与导出

- 训练完成后在 Gradio 中直接测试各 checkpoint
- 模型剪枝：去除优化器状态，从 ~5GB 压缩到 ~1.3GB
- 支持 safetensors 格式导出
- 导出的模型可直接用于推理或上传到 HuggingFace

### 3.3 部署功能 (P1 — 重要功能)

#### FR-020: Docker 容器化

- 提供官方 Docker 镜像（`ghcr.io/swivid/f5-tts:main`）
- 支持 docker-compose 部署
- GPU 透传（NVIDIA）
- HuggingFace 缓存持久化

#### FR-021: TensorRT-LLM 加速部署

- Triton Inference Server 集成
- DiT backbone 编译为 TensorRT 引擎
- 支持 FP16/BF16/FP8 量化
- 支持 Tensor Parallelism
- HTTP 和 gRPC 客户端
- 动态批处理

#### FR-022: 多硬件支持

- NVIDIA GPU (CUDA)
- AMD GPU (ROCm)
- Intel GPU (XPU / IPEX)
- Apple Silicon (MPS)
- CPU (回退方案)

### 3.4 评估功能 (P2 — 辅助功能)

#### FR-030: 质量评估

- WER (Word Error Rate): FunASR (中文) / Faster-Whisper (英文)
- SIM (说话人相似度): ECAPA-TDNN + WavLM
- UTMOS (自然度评分): SpeechMOS

#### FR-031: 批量评估

- 支持 LibriSpeech test-clean 和 Seed-TTS 测试集
- 多 GPU 分布式推理
- 可配置 NFE、ODE 方法、sway sampling 等参数

### 3.5 Python API (P0)

#### FR-040: 编程接口

```python
from f5_tts.api import F5TTS

f5tts = F5TTS(model="F5TTS_v1_Base")
wav, sr, spec = f5tts.infer(
    ref_file="ref.wav",
    ref_text="参考文本",
    gen_text="待合成文本"
)
```

- 返回 numpy 波形、采样率、mel 频谱图
- 支持导出 WAV 文件和频谱图
- 支持自动转录参考音频

---

## 4. 非功能需求

### 4.1 性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| RTF (Real-Time Factor) | < 0.05 | TensorRT-LLM 部署，L20 GPU |
| RTF (PyTorch) | < 0.15 | 标准 PyTorch 推理 |
| 首包延迟 (流式) | < 500ms | Socket Server，首块短分块优化 |
| 参考音频长度 | ≤ 12 秒 | 自动截断 |
| 单句合成上限 | ~30 秒 | 超出自动分块 |

### 4.2 质量

| 指标 | 说明 |
|------|------|
| 采样率 | 24kHz |
| Mel 通道数 | 100 |
| 说话人相似度 | 高（依赖 SIM 指标评估） |
| 语音自然度 | 依赖 UTMOS 评分 |

### 4.3 兼容性

- Python >= 3.10
- PyTorch >= 2.0.0
- 操作系统: Linux / macOS / Windows
- 硬件: NVIDIA / AMD / Intel GPU, Apple Silicon, CPU

### 4.4 可用性

- 零代码使用: Gradio Web UI 覆盖全部功能
- 一行命令推理: `f5-tts_infer-cli`
- 三行代码集成: Python API
- 容器化一键部署: Docker / docker-compose

### 4.5 安全性

- 参考音频不应泄露敏感信息
- 模型训练数据需符合数据许可协议
- 预训练模型使用 CC-BY-NC 许可（受训练数据 Emilia 限制）

---

## 5. 产品接口总览

| 接口 | 入口 | 适用场景 |
|------|------|----------|
| Gradio Web UI | `f5-tts_infer-gradio` | 交互式使用、演示、微调 |
| CLI 推理 | `f5-tts_infer-cli` | 批量生成、脚本集成 |
| CLI 微调 | `f5-tts_finetune-cli` | 多 GPU 训练、自动化流水线 |
| Gradio 微调 | `f5-tts_finetune-gradio` | 可视化微调全流程 |
| Python API | `from f5_tts.api import F5TTS` | 应用集成 |
| Socket Server | `python socket_server.py` | 实时流式传输 |
| Triton Server | `src/f5_tts/runtime/` | 生产级高并发服务 |
| Docker | `ghcr.io/swivid/f5-tts:main` | 容器化部署 |

---

## 6. 支持的语言与模型

| 语言 | 模型 | 架构 | 许可 |
|------|------|------|------|
| 中文 + 英文 | F5TTS_v1_Base | DiT (官方) | CC-BY-NC 4.0 |
| 阿拉伯语 + 英语 | silma-tts | F5-TTS Small | Apache-2.0 |
| 芬兰语 | AsmoKoskinen/F5-TTS_Finnish | F5-TTS Base | CC-BY-NC 4.0 |
| 法语 | RASPIAUDIO/F5-French | F5-TTS Base | CC-BY-NC 4.0 |
| 德语 | hvoss-techfak/F5-TTS-German | F5-TTS Base | CC-BY-NC 4.0 |
| 印地语 | SPRINGLab/F5-Hindi | F5-TTS Small | CC-BY 4.0 |
| 意大利语 | alien79/F5-TTS-italian | F5-TTS Base | CC-BY-NC 4.0 |
| 日语 | Jmica/F5TTS/JA | F5-TTS Base | CC-BY-NC 4.0 |
| 拉脱维亚语 | RaivisDejus/F5-TTS-Latvian | F5-TTS Base | CC0-1.0 |
| 俄语 | hotstone228/F5-TTS-Russian | F5-TTS Base | CC-BY-NC 4.0 |
| 西班牙语 | jpgallegoar/F5-Spanish | F5-TTS Base | CC0-1.0 |

---

## 7. 约束与限制

1. **许可限制**: 预训练模型使用 CC-BY-NC 许可，不可商用（因训练数据 Emilia 的许可要求）
2. **参考音频**: 建议长度 ≤ 12 秒，过长会自动截断
3. **单句长度**: 建议单次合成 ≤ 30 秒（自动分块可支持更长，但拼接处可能有微弱不自然）
4. **GPU 需求**: 推理最低 ~4GB 显存（Base 模型），训练需要多 GPU
5. **语言支持**: 官方模型仅支持中英文，其他语言需使用社区模型或自行微调
6. **实时性**: PyTorch 推理 RTF 约 0.15，需要 TensorRT-LLM 才能达到 0.04
