# F5-TTS 项目状态报告

**报告日期**: 2026-04-09
**当前版本**: v1.1.18
**项目状态**: 活跃开发中
**仓库**: https://github.com/SWivid/F5-TTS

---

## 1. 项目概况

### 1.1 基本信息

| 项目 | 内容 |
|------|------|
| 项目名称 | F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching |
| 当前版本 | 1.1.18 (发布于 2026-03-24) |
| 开源许可 | MIT License (代码), CC-BY-NC 4.0 (预训练模型) |
| 论文 | arXiv:2410.06885 |
| 默认模型 | F5TTS_v1_Base (DiT, 1024 dim, 22 layers) |

### 1.2 项目健康度

| 指标 | 数据 |
|------|------|
| 总提交数 | ~1000+ |
| 2025年以来提交 | 254 次 |
| 贡献者总数 | ~80+ (含社区) |
| 核心贡献者 | SWivid (248), Yushen CHEN (148), ZhikangNiu (52), lpscr (21) |
| 最近版本发布频率 | ~2-4 周一个小版本 |
| 社区语言模型 | 12 种语言，11+ 社区模型 |

---

## 2. 版本发布历史 (近期)

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| **v1.1.18** | 2026-03-24 | 分离流式/非流式推理函数，恢复并行推理支持；修复 ThreadPoolExecutor 问题 |
| **v1.1.17** | 2026-03-04 | 移除 pydantic 版本限制以兼容最新 Gradio；添加 F5TTS v1 Small + LibriTTS 训练配置 |
| **v1.1.16** | 2026-02-16 | MMDiT FlashAttention 支持；DiT text embedding 批处理优化；WandB 配置可定制化 |
| **v1.1.15** | 2026-01-19 | Gradio 6.0 升级；jieba 替换为 rjieba (Rust 绑定)；HF:// 链接支持 |
| **v1.1.12** | 2025-12 | Gradio 5.0→6.0 大版本升级，多项兼容性修复 |
| **v1.1.10** | 2025-11 | 语音编辑修复（mel 域操作，减少边界伪影） |

---

## 3. 最近开发动态 (2026 Q1)

### 3.1 已完成的重要工作

**推理优化**
- 分离流式和非流式推理函数，独立优化路径（`utils_infer.py` 重构）
- 移除无效的 `ThreadPoolExecutor`，修复并发 GPU 推理的线程安全问题
- Resampler 复用 + Vocos MelSpectrogram 实例缓存，降低训练开销
- DiT text embedding 优化为按样本批处理

**模型与配置**
- 新增 `F5TTS_v1_Small` 配置（小模型，768 dim, 18 layers）
- 新增 LibriTTS 训练配置
- `max_duration` 默认值从 4096 提升到 65536，支持更长的生成

**架构增强**
- MMDiT backbone 支持 FlashAttention
- MMDiT 支持 `torch.utils.checkpoint` 梯度检查点，降低显存
- DiT 统一 `seq_len` 命名

**训练改进**
- 训练时忽略 ground truth mel 末尾的 padding，减少无效 loss
- WandB project/run_name/resume_id 可通过 Hydra YAML 配置
- TensorBoard writer 修复，仅主进程写日志
- 修复 epoch 更新计数逻辑

**评估工具**
- `jiwer` >= 4.0.0 兼容性修复
- 新增 `ctranslate2` 安装说明

**社区生态**
- 新增阿拉伯语社区模型（SILMA AI）
- 新增拉脱维亚语社区模型
- Gradio 兼容性持续更新（pydantic 版本限制解除）

### 3.2 本地分支变更

当前本地有未提交的修改：

| 文件 | 变更 |
|------|------|
| `src/f5_tts/infer/utils_infer.py` | torchaudio 兼容性修复（9 行新增） |

---

## 4. 已知问题

### 4.1 高优先级问题

| 问题 | 位置 | 影响 | 状态 |
|------|------|------|------|
| ~~**torchcodec 硬依赖**~~ | `pyproject.toml` | 每次 `pip install -e .` 强制安装，安装后需手动卸载，否则音频解码异常 | **已修复** (代码无实际使用，从 dependencies 中移除) |
| ~~**参考音频缓存无限增长**~~ | `utils_infer.py:34-37` | `_ref_audio_cache` / `_ref_text_cache` 字典无大小限制，长时间运行导致内存泄漏 | **已修复** (OrderedDict LRU 缓存，上限 64 条，淘汰时清理临时文件) |
| ~~**多线程 GPU 推理安全隐患**~~ | `utils_infer.py` | ThreadPoolExecutor 并行调用 CUDA 存在线程安全问题 | **已修复** (commit `157cbc4` 移除 ThreadPoolExecutor，改为顺序处理) |
| ~~**音频格式支持有限**~~ | `utils_infer.py` | soundfile 不支持 MP3/M4A，用户传入这些格式直接崩溃，无友好提示 | **已修复** (`infer_process()` 添加 pydub 回退机制) |

### 4.2 中优先级问题

| 问题 | 说明 |
|------|------|
| torchaudio 依赖可移除 | 仅用于 `Resample`，可用 librosa 替代，减少一个重量级依赖 |
| Whisper 路径构建脆弱 | 4 层 `dirname` 嵌套，应使用 `pathlib.Path.resolve()` |
| Gradio 模型全量加载 | 启动时无论用不用都加载全部模型，不支持懒加载 |
| 临时文件无清理 | `preprocess_ref_audio_text()` 生成的临时 wav 不会自动删除 |

### 4.3 低优先级问题

| 问题 | 说明 |
|------|------|
| h11 控制台刷屏 | Gradio 6.x + Windows 的已知 benign 问题 |
| E2TTS 模型仍联网验证 | 本地部署时 `infer_gradio.py` 仍尝试连接 HuggingFace |

---

## 5. 性能现状

### 5.1 推理性能 (官方基准)

| 部署方式 | 硬件 | RTF | 说明 |
|----------|------|-----|------|
| TensorRT-LLM (Client-Server) | L20 GPU, 并发 2 | **0.0394** | 生产推荐 |
| TensorRT-LLM (Offline) | L20 GPU, batch=1 | **0.0402** | 离线批处理 |
| PyTorch 推理 | L20 GPU, batch=1 | **0.1467** | 标准推理 |

> RTF (Real-Time Factor): 数值越小越快。RTF=0.04 意味着生成 1 秒音频仅需 0.04 秒。

### 5.2 模型参数量

| 模型 | 架构 | 参数量级 | Checkpoint 大小 |
|------|------|----------|----------------|
| F5TTS_v1_Base | DiT, dim=1024, depth=22 | ~1.2B | ~1.3GB (剪枝后), ~5GB (完整) |
| F5TTS_v1_Small | DiT, dim=768, depth=18 | ~0.6B | 更小 |
| E2TTS_Base | UNetT, dim=1024, depth=24 | ~1.2B | ~1.3GB (剪枝后) |

---

## 6. 优化方向 (路线图)

### 6.1 短期优化 (可立即着手)

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| `torch.compile()` 集成 | Transformer 推理加速 30-50% | 中 |
| BF16 推理统一 | 更高精度，同等性能 | 低 |
| ~~LRU Cache 替换无限字典~~ | 修复内存泄漏 | ~~低~~ |
| CUDA Streams 替代 ThreadPool | 解决线程安全 + 真正 GPU 并行 | 中 |
| ~~移除 torchcodec 硬依赖~~ | 安装体验改善 | ~~低~~ |

### 6.2 中期方向

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| HTTP 微服务封装 | 更通用的部署接口 (REST API) | 中 |
| 懒加载模型 | 降低启动时间和空闲显存 | 中 |
| ~~torchcodec 移至 optional~~ | 减少默认依赖 | ~~低~~ |
| 更多社区语言模型 | 扩大用户群 | 社区驱动 |

### 6.3 长期方向

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| 更高采样率支持 (44.1kHz/48kHz) | 音频质量提升 | 高 (需重训) |
| 流式分块推理 (streaming chunk-by-chunk ODE) | 更低延迟 | 高 |
| 多模态输入 (SSML, 情感标注) | 精细化控制 | 高 |
| ONNX Runtime 部署方案 | 非 NVIDIA GPU 更好的部署选项 | 中 |

---

## 7. 社区生态

### 7.1 社区模型

12 种语言，11+ 社区贡献模型（详见 `src/f5_tts/infer/SHARED.md`）。

近期新增:
- **阿拉伯语** (SILMA AI, F5-TTS Small, Apache-2.0)
- **拉脱维亚语** (RaivisDejus, F5-TTS Base, CC0-1.0)

### 7.2 衍生项目

| 项目 | 说明 |
|------|------|
| [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx) | Apple MLX 框架实现 |
| [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) | ONNX Runtime 版本 |

### 7.3 数据集支持

官方训练数据: Emilia (95K 小时中英文)

数据准备脚本支持:
- Emilia v1/v2, WenetSpeech4TTS, LibriTTS, LJSpeech, 自定义 CSV+WAV

---

## 8. 风险与阻碍

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 许可证限制 (CC-BY-NC) | 商用受限 | 可用自有数据重新训练 |
| 训练数据依赖 (Emilia) | 数据可用性和质量决定模型质量 | 支持自定义数据集训练 |
| GPU 需求 | 推理最低 ~4GB 显存 | Small 模型减少需求；CPU 回退可用但慢 |
| ~~torchcodec 兼容性~~ | 不同平台安装问题频发 | **已移除** (代码无实际使用) |
| Gradio 版本兼容 | Gradio 大版本升级频繁导致 API 变化 | 持续跟踪更新 |

---

## 9. 里程碑时间线

```
2024-10-08  F5-TTS & E2-TTS Base 发布 (v0)
    │
2025-03-12  F5-TTS v1 Base 发布 (ConvNeXtV2 + 全头RoPE)
    │           性能大幅提升
2025-12    Gradio 6.0 迁移完成
    │
2026-01    训练管线优化 (padding loss, duration 上限, epoch 逻辑修复)
    │
2026-02    架构增强 (MMDiT FlashAttn, DiT embedding 优化, gradient checkpoint)
    │
2026-03    v1.1.18 发布 (推理分离, Small 模型配置, 社区模型扩展)
    │
2026-04    当前 (本地部署优化进行中, HTTP 微服务计划中)
    ▼
```

---

## 10. 总结

F5-TTS 项目处于**活跃且健康的开发状态**。核心模型 (F5TTS_v1_Base) 已经成熟稳定，近期工作集中在：

1. **推理优化** — 分离流式/非流式路径、并行推理修复
2. **模型扩展** — Small 模型、更多训练配置
3. **社区生态** — 持续增长的语言覆盖（12+ 语言）
4. **部署体验** — Docker、TensorRT-LLM 加速

主要技术债务集中在 `utils_infer.py` 的缓存管理、线程安全和依赖清理上。这些是可解决的中等复杂度问题，不影响核心功能使用。
