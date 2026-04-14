# F5-TTS 项目状态报告

**报告日期**: 2026-04-10
**当前版本**: v1.1.18
**项目状态**: 活跃开发中
**仓库**: https://github.com/SWivid/F5-TTS

---

## 1. 项目概况

| 项目 | 内容 |
|------|------|
| 项目名称 | F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching |
| 当前版本 | 1.1.18 (发布于 2026-03-24) |
| 开源许可 | MIT License (代码), CC-BY-NC 4.0 (预训练模型) |
| 论文 | arXiv:2410.06885 |
| 默认模型 | F5TTS_v1_Base (DiT, dim=1024, depth=22, heads=16) |
| Vocoder | Vocos (默认) / BigVGAN |
| 音频输出 | 24kHz 采样率, 100 mel 通道, hop_length=256, n_fft=1024 |

---

## 2. 技术架构

### 2.1 核心数据流

```
文本输入 → Tokenizer (pinyin/char/byte/custom) → Text Embedding
                                                    ↓
参考音频 → Mel 频谱提取 → Flow Matching (CFM) → 预测 Mel → Vocoder → 音频波形
```

**推理流程**: 预处理参考音频(去静音/截断/重采样) → 文本分块 → 逐块 CFM.sample() ODE 求解 → Vocoder 解码 → 交叉淡入淡出拼接

**训练流程**: 数据集 → 按帧数动态分批 → 随机 span mask → MSE loss (仅 mask 区域) → EMA 权重更新

### 2.2 支持的 Backbone 架构

| 架构 | 类 | 文件 | 特性 | 对应模型 |
|------|----|------|------|----------|
| DiT | `DiT` | `model/backbones/dit.py` | AdaLN + RoPE + ConvNeXtV2 | F5TTS_v1_Base/Small, F5TTS_Base/Small |
| UNetT | `UNetT` | `model/backbones/unett.py` | RMSNorm + skip connections (add/concat) | E2TTS_Base/Small |
| MMDiT | `MMDiT` | `model/backbones/mmdit.py` | SD3 风格双流: 音频/文本独立流 + JointAttention | 可用但无预训练配置 |

### 2.3 模型配置

| 配置文件 | Backbone | dim | depth | heads | ff_mult | text_dim | 分词器 |
|----------|----------|-----|-------|-------|---------|----------|--------|
| `F5TTS_v1_Base.yaml` | DiT | 1024 | 22 | 16 | 2 | 512 | pinyin |
| `F5TTS_v1_Small.yaml` | DiT | 768 | 18 | 12 | 2 | 512 | char |
| `F5TTS_Base.yaml` | DiT | 1024 | 22 | 16 | 2 | 512 | pinyin |
| `F5TTS_Small.yaml` | DiT | 768 | 18 | 12 | 2 | 512 | pinyin |
| `E2TTS_Base.yaml` | UNetT | 1024 | 24 | 16 | 4 | — | pinyin |
| `E2TTS_Small.yaml` | UNetT | 768 | 20 | 12 | 4 | — | pinyin |

### 2.4 分词器

| 分词器 | 说明 | 词表来源 |
|--------|------|----------|
| `pinyin` | 中文拼音 + 声调（默认） | 数据集拼音词表 |
| `char` | 字符级分词 | 数据集字符词表 |
| `byte` | UTF-8 字节分词 (ByT5 风格) | 固定 256 |
| `custom` | 用户自定义词表文件 | 用户指定路径 |

> 注: `byte` 分词器在核心代码中可用，但 CLI 和 Gradio 微调界面未暴露该选项。

### 2.5 推理精度

模型加载时自动选择精度（`load_checkpoint()`, `utils_infer.py`）:

| 条件 | dtype | 说明 |
|------|-------|------|
| CUDA + compute capability >= 7 + 非 ZLUDA | FP16 | Tensor Core 加速 |
| 其他 (MPS, XPU, CPU, ZLUDA) | FP32 | 全精度回退 |

RMSNorm 中已有 FP32 上转保护（`modules.py`），防止半精度下溢。训练支持 `none/fp16/bf16` 混合精度选项。

### 2.6 硬件支持

自动检测优先级: CUDA → XPU (Intel) → MPS (Apple Silicon) → CPU

---

## 3. 入口点与常用命令

### 入口点 (`pyproject.toml [project.scripts]`)

| 命令 | 函数 |
|------|------|
| `f5-tts_infer-cli` | `f5_tts.infer.infer_cli:main` |
| `f5-tts_infer-gradio` | `f5_tts.infer.infer_gradio:main` |
| `f5-tts_finetune-cli` | `f5_tts.train.finetune_cli:main` |
| `f5-tts_finetune-gradio` | `f5_tts.train.finetune_gradio:main` |

### 常用命令

```bash
# 安装
pip install -e .                  # 本地可编辑安装 (训练 + 推理)
pip install f5-tts                # pip 安装 (仅推理)

# 推理
f5-tts_infer-gradio --port 7860   # Gradio Web UI
f5-tts_infer-cli -c config.toml   # CLI 推理

# 微调训练
f5-tts_finetune-cli               # CLI 微调
f5-tts_finetune-gradio            # Gradio 微调 UI

# 完整训练
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml
```

---

## 4. 依赖

### 核心依赖 (25 个)

`torch>=2.0.0`, `torchaudio>=2.0.0`, `transformers`, `torchdiffeq`, `vocos`, `x_transformers>=1.31.14`, `ema_pytorch>=0.5.2`, `accelerate>=0.33.0`, `hydra-core>=1.3.0`, `gradio>=6.0.0,<6.11.0`, `librosa`, `soundfile`, `pydub`, `cached_path`, `datasets`, `click`, `tqdm>=4.65.0`, `safetensors`, `tomli`, `wandb`, `matplotlib`, `pypinyin`, `rjieba`, `unidecode`, `bitsandbytes>0.37.0` (非 arm64/Darwin)

> 注: `gradio` 限制 `<6.11.0` 是因为 6.11.0 存在 tab 切换冻结 bug (gradio-app/gradio#13198)。

### 评估依赖 (可选, `pip install f5-tts[eval]`)

`faster_whisper==0.10.1`, `funasr`, `jiwer`, `modelscope`, `zhconv`, `zhon`

### 已移除的依赖

| 包 | 移除原因 |
|----|----------|
| `torchcodec` | 代码中无实际使用，属于多余硬依赖，强制安装后导致音频解码异常 |

---

## 5. 已修复的问题

### 5.1 本地部署修复 (未提交上游)

| 问题 | 根因 | 修复方案 | Commit |
|------|------|----------|--------|
| torchcodec 硬依赖 | 上游 pyproject.toml 列入 dependencies，但代码无任何 import 或使用 | 从 dependencies 移除 | `6e2c09f` |
| 参考音频缓存无限增长 | `_ref_audio_cache` / `_ref_text_cache` 使用普通 dict，无淘汰机制 | OrderedDict LRU 缓存，上限 64 条 | `6e2c09f` |
| 模型路径构建脆弱 | 3 处使用 4 层 `os.path.dirname` 嵌套定位项目根目录 | `Path(__file__).resolve().parents[3]`，提升为模块级常量 | `9ff7dee` |
| Gradio 6.11.0 tab 冻结 | Gradio 6.11.0 的 tab 切换触发全量重渲染 | 限制 gradio<6.11.0，优化 Multi-Speech 组件数量 | `38983de` |
| 重新上传音频后文本未清空 | Gradio 界面未在文件变更时清空关联文本输入 | 添加 change handler 清空 ref_text | `7a8ec43` |
| 多线程推理张量尺寸不匹配 | ThreadPoolExecutor 并行调用 CUDA，DiT text embedding 缓存被覆盖 | 移除 ThreadPoolExecutor，改为顺序处理 | `157cbc4` |
| 音频格式支持有限 | soundfile 不支持 MP3/M4A，直接崩溃 | `infer_process()` 添加 pydub 回退机制 | `810482f` |

### 5.2 上游已修复 (v1.1.12 ~ v1.1.18)

- 分离流式和非流式推理函数 (`utils_infer.py` 重构)
- Resampler 复用 + Vocos MelSpectrogram 实例缓存
- DiT text embedding 优化为按样本批处理
- 训练时忽略 ground truth mel 末尾的 padding
- WandB project/run_name/resume_id 可通过 Hydra YAML 配置
- `jiwer` >= 4.0.0 兼容性修复
- MMDiT backbone 支持 FlashAttention 和梯度检查点
- `max_duration` 默认值从 4096 提升到 65536

---

## 6. 已知问题

### 6.1 需关注

| 问题 | 位置 | 影响 | 建议 |
|------|------|------|------|
| 临时 wav 文件无主动清理 | `utils_infer.py` | LRU 缓存上限 64 条约束了文件数量（约 6-32MB），但淘汰时不删除临时文件（避免多线程竞态），依赖 OS 清理 | 可接受，后续可加引用计数 |
| Gradio 模型全量加载 | `infer_gradio.py` | Vocoder + F5TTS 模型在模块导入时立即加载（~1.5GB 显存），不支持懒加载 | 多模型切换或资源受限部署时再优化 |
| E2TTS 模型仍联网验证 | `infer_gradio.py` | 本地部署时仍尝试连接 HuggingFace 验证 E2TTS 模型 | 低优先级 |

### 6.2 已评估但不建议修改

| 问题 | 评估结论 |
|------|----------|
| torchaudio 依赖可移除 | 深度集成于 15 个文件 47 处调用（Resample/load/save/MelSpectrogram/info），替换成本高且收益低 |
| `torch.compile()` 集成 | 动态形状导致 recompilation，DiT cache 机制与 Dynamo 冲突，生产部署已有 TensorRT-LLM 方案 |
| BF16 推理统一 | 已有自动 FP16/FP32 选择逻辑，RMSNorm 有 FP32 上转保护，BF16 需 Ampere+ 硬件且收益有限 |

### 6.3 已知 benign 问题

| 问题 | 说明 |
|------|------|
| h11 控制台刷屏 | Gradio 6.x + Windows 的已知问题，不影响功能 |

---

## 7. 性能现状

### 7.1 推理性能 (官方基准, L20 GPU)

| 部署方式 | RTF | 说明 |
|----------|-----|------|
| TensorRT-LLM (Client-Server, 并发 2) | **0.039** | 生产推荐 |
| TensorRT-LLM (Offline, batch=1) | **0.040** | 离线批处理 |
| PyTorch (batch=1) | **0.147** | 标准推理 |

> RTF (Real-Time Factor): 数值越小越快。RTF=0.04 意味着生成 1 秒音频仅需 0.04 秒。

### 7.2 模型参数量

| 模型 | 架构 | 参数量 | Checkpoint 大小 |
|------|------|--------|----------------|
| F5TTS_v1_Base | DiT, dim=1024, depth=22 | ~1.2B | ~1.3GB (EMA 剪枝后) |
| F5TTS_v1_Small | DiT, dim=768, depth=18 | ~0.6B | 更小 |
| E2TTS_Base | UNetT, dim=1024, depth=24 | ~1.2B | ~1.3GB (EMA 剪枝后) |

---

## 8. 社区生态

### 8.1 社区模型 (10 种语言)

| 语言 | 模型规格 | 许可证 |
|------|----------|--------|
| 阿拉伯语 (+ 英语) | F5-TTS Small | Apache-2.0 |
| 芬兰语 | F5-TTS Base | CC-BY-NC-4.0 |
| 法语 | F5-TTS Base | CC-BY-NC-4.0 |
| 德语 | F5-TTS Base | CC-BY-NC-4.0 |
| 印地语 | F5-TTS Small | CC-BY-4.0 |
| 意大利语 | F5-TTS Base | CC-BY-NC-4.0 |
| 日语 | F5-TTS Base | CC-BY-NC-4.0 |
| 拉脱维亚语 | F5-TTS Base | CC0-1.0 |
| 俄语 | F5-TTS Base | CC-BY-NC-4.0 |
| 西班牙语 | F5-TTS Base | CC0-1.0 |

详细模型列表见 `src/f5_tts/infer/SHARED.md`。

### 8.2 衍生项目

| 项目 | 说明 |
|------|------|
| [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) | Apple MLX 框架实现 |
| [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) | ONNX Runtime 版本 |

### 8.3 数据集支持

官方训练数据: Emilia (95K 小时中英文)

数据准备脚本支持: Emilia v1/v2, WenetSpeech4TTS, LibriTTS, LJSpeech, 自定义 CSV+WAV

---

## 9. 优化方向

### 9.1 可行

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| CUDA Streams 替代顺序处理 | GPU 并行推理（当前为顺序） | 中 |
| HTTP 微服务封装 | 更通用的部署接口 (REST API) | 中 |
| 懒加载模型 | 降低启动时间和空闲显存 | 中 |

### 9.2 长期方向

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| 更高采样率支持 (44.1kHz/48kHz) | 音频质量提升 | 高 (需重训) |
| 流式分块推理 (streaming chunk-by-chunk ODE) | 更低延迟 | 高 |
| 多模态输入 (SSML, 情感标注) | 精细化控制 | 高 |
| ONNX Runtime 部署方案 | 非 NVIDIA GPU 更好的部署选项 | 中 |
| 更多社区语言模型 | 扩大用户群 | 社区驱动 |

---

## 10. 风险与阻碍

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 许可证限制 (CC-BY-NC) | 商用受限 | 可用自有数据重新训练 |
| 训练数据依赖 (Emilia) | 数据可用性和质量决定模型质量 | 支持自定义数据集训练 |
| GPU 需求 | 推理最低 ~4GB 显存 | Small 模型减少需求；CPU 回退可用但慢 |
| Gradio 版本兼容 | Gradio 大版本升级频繁导致 API 变化 | 持续跟踪更新 |

---

## 11. 里程碑时间线

```
2024-10-08  F5-TTS & E2-TTS Base 发布 (v0)
    |
2025-03-12  F5-TTS v1 Base 发布 (ConvNeXtV2 + 全头 RoPE, 性能大幅提升)
    |
2025-12     Gradio 6.0 迁移完成
    |
2026-01     训练管线优化 (padding loss, duration 上限, epoch 逻辑修复)
    |
2026-02     架构增强 (MMDiT FlashAttn, DiT embedding 优化, gradient checkpoint)
    |
2026-03     v1.1.18 发布 (推理分离, Small 模型配置, 社区模型扩展)
    |
2026-04     本地部署优化 (依赖清理, 缓存修复, 路径重构)
    v
```

---

## 12. 文档索引

| 文档 | 路径 |
|------|------|
| 产品需求文档 (PRD) | `docs/Product_Requirement_Document.md` |
| 系统设计文档 (SDD) | `docs/System_Design_Document.md` |
| FAQ | `docs/F5-TTS_FAQ.md` |
| 推理使用指南 | `src/f5_tts/infer/README.md` |
| 训练使用指南 | `src/f5_tts/train/README.md` |
| 社区模型列表 | `src/f5_tts/infer/SHARED.md` |
| Triton 部署指南 | `src/f5_tts/runtime/triton_trtllm/README.md` |
| 部署指南 | `DEPLOY-GUIDE.md` |
