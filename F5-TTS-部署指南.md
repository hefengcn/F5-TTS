# F5-TTS 部署指南

> 在 Windows 11 + RTX 5060 Laptop (8GB VRAM) 环境下从源码部署 F5-TTS 的实战指南。
> 包含部署步骤、踩坑记录和日常使用方法。

---

## 目录

1. [环境信息](#1-环境信息)
2. [部署步骤](#2-部署步骤)
3. [日常使用](#3-日常使用)
4. [验证](#4-验证)
5. [踩坑记录](#5-踩坑记录)
6. [常见问题](#6-常见问题)

---

## 1. 环境信息

### 硬件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 5060 Laptop, 8GB VRAM |
| CPU | Intel i7-13700HX (16核24线程) |
| 内存 | 16 GB DDR5 |
| 存储 | NVMe SSD 1TB |

### 软件栈

| 组件 | 版本 |
|------|------|
| 操作系统 | Windows 11 Home China |
| Python | 3.10.20 (conda) |
| PyTorch | 2.10.0+cu128 |
| torchaudio | 2.10.0+cu128 |
| CUDA | 12.8 |
| Gradio | 6.10.0 |
| conda 环境 | `f5-tts-env`，路径 `D:\CondaEnv\f5-tts-env` |

### 与 Index-TTS 对比

| 对比项 | Index-TTS | F5-TTS |
|--------|-----------|--------|
| 架构 | 自回归 GPT | 非自回归扩散 Transformer |
| 包管理 | UV | pip |
| 入口 | `webui.py` | `src/f5_tts/infer/infer_gradio.py` |
| 默认端口 | 7860 | 7860 |
| CUDA | 12.8 (cu128) | 12.8 (cu128) |
| PyTorch | 2.10+cu128 | 2.10+cu128 |
| 情感控制 | 支持 | 不支持（纯音色克隆） |

> 两个项目使用独立的 conda 环境，互不干扰。

---

## 2. 部署步骤

### 步骤 1：克隆源码

```bash
cd d:\Workspace\MyAIModels\F5-TTS
git clone https://github.com/SWivid/F5-TTS.git .
```

> 如果目标目录非空，先克隆到临时目录再移入（包括 `.git` 目录）。

### 步骤 2：从 Index-TTS 克隆 Conda 环境

```bash
conda create -p D:\CondaEnv\f5-tts-env --clone index-tts
conda config --add envs_dirs D:\CondaEnv
conda activate f5-tts-env
```

> 从已部署成功的 Index-TTS 环境克隆，省去单独安装 PyTorch/CUDA 的步骤。

### 步骤 3：安装 F5-TTS 依赖

```bash
cd d:\Workspace\MyAIModels\F5-TTS
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**必须卸载 torchcodec**（安装依赖时会被自动拉入，Windows 上缺少 FFmpeg 共享库会导致崩溃）：

```bash
pip uninstall torchcodec -y
```

> **注意**：每次重新执行 `pip install -e .` 后都需要再次卸载 torchcodec。

### 步骤 4：配置 HuggingFace 镜像（国内网络必需）

```bash
# 当前 shell 临时生效
export HF_ENDPOINT=https://hf-mirror.com

# 永久生效：写入 conda 环境激活脚本
mkdir -p D:/CondaEnv/f5-tts-env/etc/conda/activate.d
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> D:/CondaEnv/f5-tts-env/etc/conda/activate.d/env_vars.sh
```

### 步骤 5：下载预训练模型

**TTS 模型**（从魔塔社区下载，约 1.26 GB）：

```bash
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "from modelscope import snapshot_download; snapshot_download('SWivid/F5-TTS_Emilia-ZH-EN', local_dir='ckpts/F5TTS_v1_Base')"
```

**Vocos 声码器**（从 HuggingFace 镜像下载）：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('charactr/vocos-mel-24khz', 'config.yaml', local_dir='ckpts/vocos-mel-24khz')
hf_hub_download('charactr/vocos-mel-24khz', 'pytorch_model.bin', local_dir='ckpts/vocos-mel-24khz')
print('Done!')
"
```

**Whisper ASR 模型**（用于自动转录参考音频，可选但推荐）：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/whisper-large-v3-turbo', local_dir='ckpts/whisper-large-v3-turbo')
print('Done!')
"
```

下载后的目录结构：

```
ckpts/
├── F5TTS_v1_Base/
│   ├── model_1250000.safetensors   (~1.26 GB)
│   ├── vocab.txt
│   └── ...
├── vocos-mel-24khz/
│   ├── config.yaml
│   └── pytorch_model.bin
├── whisper-large-v3-turbo/
│   ├── model.safetensors
│   └── ...
```

> 不提前下载的话，部分模型会在首次启动时自动下载（国内网络可能极慢）。

### 步骤 6：启动 Gradio Web UI

```bash
conda activate f5-tts-env
cd d:\Workspace\MyAIModels\F5-TTS
python src/f5_tts/infer/infer_gradio.py
```

启动后浏览器访问 `http://127.0.0.1:7860`。

---

## 3. 日常使用

### 启动服务

见上方 [步骤 6](#步骤-6启动-gradio-web-ui)。控制台可能刷 h11 错误，这是 Gradio 6.x 在 Windows 上的 benign 问题，不影响使用。

### CLI 推理

```bash
f5-tts_infer-cli --ref_audio "参考音频.wav" --ref_text "参考音频的文字" --gen_text "要合成的文本" -o . -w "输出.wav"
```

### Python API

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    ckpt_file="ckpts/F5TTS_v1_Base/model_1250000.safetensors",
    vocab_file="ckpts/F5TTS_v1_Base/vocab.txt",
    vocoder_local_path="ckpts/vocos-mel-24khz",
)
wav, sr, spec = tts.infer(
    ref_file="参考音频.wav",
    ref_text="参考音频文字",
    gen_text="要合成的文本",
    file_wave="输出.wav",
)
```

---

## 4. 验证

1. **CUDA 验证**:
   ```bash
   conda activate f5-tts-env
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   → 应输出 `True NVIDIA GeForce RTX 5060 Laptop GPU`

2. **依赖验证**: `python -c "from f5_tts import infer; print('OK')"` → 无报错

3. **UI 验证**: 浏览器打开 `http://localhost:7860`，上传参考音频 + 文字，输入合成文本，点击生成

4. **推理验证**: 生成约 10 秒音频，确认音质和克隆效果正常

---

## 5. 踩坑记录

### torchcodec 强依赖问题

**症状**：推理或 ASR 时报 `RuntimeError: Could not load libtorchcodec` 或 `ImportError`。

**原因**：
- F5-TTS 的 `pyproject.toml` 依赖了 `torchcodec`
- torchaudio 2.10 的 `load()` 硬编码调用 torchcodec，忽略 `backend` 参数
- torchcodec 需要 FFmpeg **共享库** (`.dll`)，Windows 上通常只有静态版 (`.exe`)
- `transformers` 的 ASR pipeline 也会 `import torchcodec`

**解决**：代码已修改为使用 soundfile 替代 torchaudio.load()，推理不再触发 torchcodec。建议卸载以避免 transformers ASR pipeline 调用时崩溃：

```bash
pip uninstall torchcodec -y
```

### 模型下载极慢

**原因**：国内直连 HuggingFace 几乎不可用。

**解决**：
- TTS 主模型：从魔塔社区 (ModelScope) 下载
- Vocos / Whisper：使用 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载

### 代码默认从 HuggingFace 加载模型

代码已修改为从本地 `ckpts/` 目录加载模型，无需联网验证。如需恢复远程加载，查看 `src/f5_tts/infer/infer_gradio.py` 中的 `_DEFAULT_CKPT_DIR` 配置。

### Gradio 6.x h11 报错

**症状**：控制台大量 `h11._util.LocalProtocolError: Too little data for declared Content-Length`。

**影响**：无。Web UI 正常工作，只是控制台输出噪音。

### 长文本推理 tensor 尺寸不匹配

**症状**：`RuntimeError: Sizes of tensors must match`。

**原因**：F5-TTS 用 `ThreadPoolExecutor` 并行推理，多 batch 音频长度差异大时 `torch.cat` 失败。

**解决**：单次生成不超过 30 秒（8GB 显存），长文本分多次生成。

### conda 环境损坏重建

```bash
conda remove -p D:\CondaEnv\f5-tts-env --all
conda create -p D:\CondaEnv\f5-tts-env --clone index-tts
conda activate f5-tts-env
pip install -e .
pip uninstall torchcodec -y
```

本地模型文件在 `ckpts/` 目录中不受影响。

---

## 6. 常见问题

### Q: pip install 很慢怎么办？

```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或永久配置：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: CUDA out of memory 怎么办？

- 8GB 显存通常足够生成 30 秒以内的音频
- 确保没有其他 GPU 占用程序（如 Index-TTS 同时运行）

### Q: git pull 后本地修改被覆盖？

```bash
git stash          # 暂存本地修改
git pull           # 拉取更新
git stash pop      # 恢复本地修改
```

### Q: RTX 5060 驱动要求？

需要 NVIDIA 驱动 570+，已通过 Index-TTS 验证驱动正常。

---

## 附录 A：国内镜像源

| 用途 | 镜像地址 |
|------|---------|
| PyPI (pip) | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| HuggingFace | `https://hf-mirror.com` |
| 魔塔社区 (ModelScope) | `https://www.modelscope.cn` |
| Conda (conda-forge) | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` |

## 附录 B：核心依赖版本

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.10.0+cu128 | 与 CUDA 12.8 匹配 |
| torchaudio | 2.10.0+cu128 | 强制依赖 torchcodec，代码已绕过 |
| torchcodec | 0.11.0 (建议卸载) | Windows 缺 FFmpeg 共享库，无法加载；代码已绕过 |
| gradio | 6.10.0 | h11 错误是 benign |
| numpy | 1.26.2 | — |
