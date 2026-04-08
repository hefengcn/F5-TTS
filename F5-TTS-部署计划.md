# F5-TTS 部署计划

## Context

在 `d:\Workspace\MyAIModels\F5-TTS` 目录下从源码部署 F5-TTS，使用 Gradio Web UI 模式运行。

### 用户硬件

| 项目 | 配置 |
|------|------|
| GPU | RTX 5060 Laptop, 8GB VRAM |
| CPU | i7-13700HX (16核24线程) |
| 内存 | 16 GB |
| 存储 | NVMe SSD 1TB |

### 参考环境（Index-TTS 已部署成功）

| 组件 | Index-TTS 版本 | F5-TTS 对齐版本 |
|------|---------------|----------------|
| Python | 3.10.20 | 3.10 |
| PyTorch | 2.10.0+cu128 | 2.10+cu128 |
| torchaudio | 2.10.0+cu128 | 2.10+cu128 |
| CUDA | 12.8 | 12.8 |
| numpy | 1.26.2 | 复用 |
| transformers | 4.52.1 | 复用 |
| accelerate | 1.8.1 | 复用 |
| gradio | 5.45.0 | 复用 |
| conda env | `index-tts` | `f5-tts`（独立环境） |

---

## 步骤 1：克隆 F5-TTS 源码

```bash
cd d:\Workspace\MyAIModels\F5-TTS
git clone https://github.com/SWivid/F5-TTS.git .
```

## 步骤 2：创建 Conda 虚拟环境

```bash
conda create -n f5-tts python=3.10 -y
conda activate f5-tts
```

> 使用 Python 3.10，与 Index-TTS 保持一致，确保 CUDA 12.8 + PyTorch 2.10 兼容。

## 步骤 3：安装 PyTorch（CUDA 12.8，与 Index-TTS 一致）

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 与 Index-TTS 使用相同的 CUDA 12.8 版本，确保 RTX 5060 兼容性。

## 步骤 4：安装 F5-TTS 依赖

```bash
cd d:\Workspace\MyAIModels\F5-TTS
pip install -e ".[infer]"
```

> 使用 `-e` 可编辑模式安装，方便后续修改源码。`[infer]` 仅安装推理所需依赖（含 Gradio）。

## 步骤 5：设置 HuggingFace 镜像（国内网络必需）

在安装 F5-TTS 依赖之前/之后设置镜像环境变量，加速模型和依赖下载：

```bash
# Windows Git Bash（conda activate 之后在当前 shell 生效）
export HF_ENDPOINT=https://hf-mirror.com

# 如需永久生效，写入 conda 环境激活脚本：
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/miniconda3/envs/f5-tts/etc/conda/activate.d/env_vars.sh
mkdir -p ~/miniconda3/envs/f5-tts/etc/conda/activate.d
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/miniconda3/envs/f5-tts/etc/conda/activate.d/env_vars.sh
```

## 步骤 6：下载预训练模型（可选，也可首次启动时自动下载）

```bash
huggingface-cli download SWivid/F5-TTS_Base --local-dir ./ckpts/F5-TTS_Base
```

> 模型约 1.2GB。不提前下载的话，首次运行 Gradio UI 时会自动下载。

## 步骤 7：启动 Gradio Web UI

```bash
conda activate f5-tts
cd d:\Workspace\MyAIModels\F5-TTS
python src/f5_tts/infer/gradio_api.py
```

启动后浏览器访问 `http://localhost:7860`。

---

## 关键文件（克隆后）

| 文件 | 作用 |
|------|------|
| `src/f5_tts/infer/gradio_api.py` | Gradio Web UI 入口 |
| `src/f5_tts/infer/cli_infer.py` | CLI 推理入口 |
| `src/f5_tts/infer/api.py` | FastAPI 服务入口（可选启用） |
| `src/f5_tts/infer/utils_infer.py` | 推理工具函数 |
| `pyproject.toml` | 项目依赖配置 |

---

## 验证步骤

1. **CUDA 验证**:
   ```bash
   conda activate f5-tts
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   → 应输出 `True NVIDIA GeForce RTX 5060 Laptop GPU`

2. **依赖验证**: `python -c "from f5_tts import infer; print('OK')"` → 无报错

3. **UI 验证**: 启动 Gradio 后浏览器打开 `http://localhost:7860`，上传一段参考音频 + 对应文字，输入要合成的文本，点击生成

4. **推理验证**: 生成一段 10 秒左右的音频，确认音质和克隆效果正常

---

## 与 Index-TTS 的对比

| 对比项 | Index-TTS | F5-TTS |
|--------|-----------|--------|
| 架构 | 自回归 GPT | 非自回归扩散 Transformer |
| 包管理 | UV (pyproject.toml) | pip (setup.cfg / pyproject.toml) |
| Python 入口 | `webui.py` | `src/f5_tts/infer/gradio_api.py` |
| 默认端口 | 7860 | 7860 |
| CUDA 版本 | 12.8 (cu128) | 12.8 (cu128) ← 对齐 |
| PyTorch | 2.10+cu128 | 2.10+cu128 ← 对齐 |
| 情感控制 | 支持 | 不支持（纯音色克隆） |
| 长文本 | 支持较好 | 长文本可能需分段 |

> **注意**：两个项目使用独立的 conda 环境，互不干扰。

---

## 注意事项

- RTX 5060 需要最新 NVIDIA 驱动（建议 570+），您已通过 Index-TTS 验证驱动正常
- 8GB 显存足够运行 F5-TTS，长文本（>30秒）可能需要分段生成
- F5-TTS 使用 pip 安装（不像 Index-TTS 使用 UV），两者包管理方式不同
- 如果遇到依赖冲突，可以在 F5-TTS 环境中单独 `pip install` 冲突的包
- 模型首次加载需要约 1-2 分钟，后续推理速度取决于文本长度和 GPU 性能
