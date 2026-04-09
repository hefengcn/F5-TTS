# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

F5-TTS 是一个基于 Flow Matching + Diffusion Transformer (DiT) 的开源文本转语音 (TTS) 系统。核心能力是**零样本语音克隆**：仅需一段参考音频即可合成该说话人的语音，无需微调。同时支持微调训练以适配特定说话人或新语言。

- 音频输出: 24kHz 采样率，100 通道 mel 频谱
- Vocoder: Vocos (默认) 或 BigVGAN
- 支持架构: F5-TTS (DiT + ConvNeXtV2)、E2-TTS (Flat-UNet Transformer)、MMDiT (SD3-style 双流)
- 默认模型: F5TTS_v1_Base (dim=1024, depth=22, heads=16)
- 当前版本: v1.1.18

## 常用命令

### 安装
```bash
pip install -e .                  # 本地可编辑安装 (训练 + 推理)
pip install f5-tts                # pip 安装 (仅推理)
```

### 推理
```bash
f5-tts_infer-gradio --port 7860 --host 0.0.0.0   # Gradio Web UI
f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio ref.wav --ref_text "文本" --gen_text "待合成文本"  # CLI 推理
f5-tts_infer-cli -c config.toml                    # 使用 TOML 配置文件推理
```

### 训练
```bash
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml  # 完整训练
f5-tts_finetune-cli     # CLI 微调
f5-tts_finetune-gradio  # Gradio 微调 UI
```

### 代码质量
```bash
pre-commit run --all-files   # 运行 linter (ruff) + formatter + import sorter + YAML 检查
```
代码检查工具为 **ruff** (v0.11.2)。部分模型组件因张量表示需要，有 E722 linting 例外。

## 架构

### 核心数据流
```
文本输入 → Tokenizer (pinyin/char/byte/custom) → Text Embedding
                                                    ↓
参考音频 → Mel 频谱提取 → Flow Matching (CFM) → 预测 Mel → Vocoder → 音频波形
```

**推理流程**: 预处理参考音频(去静音/截断/重采样) → 文本分块 → 逐块 CFM.sample() ODE 求解 → Vocoder 解码 → 交叉淡入淡出拼接

**训练流程**: 数据集 → 按帧数动态分批 → 随机 span mask → MSE loss (仅 mask 区域) → EMA 权重更新

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| Flow Matching 核心 | `model/cfm.py` | CFM 算法实现: 训练前向传播、ODE 采样、CFG、sway sampling |
| DiT Backbone | `model/backbones/dit.py` | F5-TTS 主架构: AdaLN + RoPE + ConvNeXtV2 层 |
| UNetT Backbone | `model/backbones/unett.py` | E2-TTS 架构: RMSNorm + skip connections |
| MMDiT Backbone | `model/backbones/mmdit.py` | SD3 风格双流架构: 音频/文本独立流 + JointAttention |
| 构建模块 | `model/modules.py` | MelSpec, ConvNeXtV2, AdaLayerNorm, DiTBlock, Attention 等 |
| 推理工具 | `infer/utils_infer.py` | 模型/声码器加载、分块推理、交叉淡入淡出、流式推理 |
| Python API | `api.py` | `F5TTS` 类: 高层推理接口 |
| 训练器 | `model/trainer.py` | Accelerate 分布式训练、EMA、WandB/TensorBoard |
| 数据管线 | `model/dataset.py` | HF/Arrow 数据集加载、DynamicBatchSampler |
| 流式服务 | `socket_server.py` | TCP 实时流式推理 |
| 生产部署 | `runtime/triton_trtllm/` | Triton + TensorRT-LLM 加速部署 |

### 配置系统

- **Hydra + OmegaConf** YAML 配置，位于 `src/f5_tts/configs/`
- 模型变体: `F5TTS_v1_Base` (dim=1024, depth=22)、`F5TTS_v1_Small` (dim=768, depth=18)、`F5TTS_Base`、`E2TTS_Base`/`E2TTS_Small`
- 每个配置定义: 模型架构、mel 频谱参数、训练超参、数据集设置、checkpoint/日志
- Backbone 通过 `hydra.utils.get_class` 动态实例化（可插拔）

### 入口点 (pyproject.toml `[project.scripts]`)

| 命令 | 函数 |
|------|------|
| `f5-tts_infer-cli` | `f5_tts.infer.infer_cli:main` |
| `f5-tts_infer-gradio` | `f5_tts.infer.infer_gradio:main` |
| `f5-tts_finetune-cli` | `f5_tts.train.finetune_cli:main` |
| `f5-tts_finetune-gradio` | `f5_tts.train.finetune_gradio:main` |

### 硬件支持

自动检测优先级: CUDA → XPU (Intel) → MPS (Apple Silicon) → CPU

## 项目约定

- 包源码在 `src/f5_tts/` (src layout)
- 配置使用 Hydra YAML，支持 `${...}` 变量插值
- 音频参数: 24kHz 采样率、100 mel 通道、hop_length 256、n_fft 1024
- 默认分词器: `pinyin` (中文拼音+声调)；支持 `char`、`byte`、`custom` vocab 文件
- 推理使用 EMA 权重（非训练权重）
- 评估依赖可选: `pip install f5-tts[eval]`
- 社区模型注册在 `src/f5_tts/infer/SHARED.md`

## 已知技术债务

详见 `docs/Project_Status_Report.md`，关键项：

- **torchcodec 硬依赖**: pyproject.toml 强制安装，安装后需手动卸载
- **参考音频缓存无限增长**: `utils_infer.py` 的 `_ref_audio_cache` 无大小限制，需替换为 LRU Cache
- **多线程 GPU 推理安全隐患**: ThreadPoolExecutor 并行 CUDA 调用存在线程安全问题

## 文档索引

| 文档 | 路径 |
|------|------|
| 产品需求文档 (PRD) | `docs/Product_Requirement_Document.md` |
| 系统设计文档 (SDD) | `docs/System_Design_Document.md` |
| 项目状态报告 | `docs/Project_Status_Report.md` |
| 推理使用指南 | `src/f5_tts/infer/README.md` |
| 训练使用指南 | `src/f5_tts/train/README.md` |
| 社区模型列表 | `src/f5_tts/infer/SHARED.md` |
| Triton 部署指南 | `src/f5_tts/runtime/triton_trtllm/README.md` |
