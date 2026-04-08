# F5-TTS 部署指南（实战总结）

> 本文档记录了在 Windows 11 + NVIDIA RTX 5060 Laptop (8GB VRAM) 环境下，
> 从零开始部署 F5-TTS 的完整步骤、踩坑记录和解决方案。
>
> 最后更新：2026-04-08

---

## 目录

1. [环境信息](#1-环境信息)
2. [部署步骤](#2-部署步骤)
3. [踩坑记录与解决方案](#3-踩坑记录与解决方案)
4. [已做的代码修改](#4-已做的代码修改)
5. [日常使用](#5-日常使用)
6. [常见问题](#6-常见问题)

---

## 1. 环境信息

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 Home China |
| GPU | NVIDIA GeForce RTX 5060 Laptop, 8GB VRAM |
| CPU | Intel i7-13700HX (16核24线程) |
| 内存 | 16 GB DDR5 |
| 存储 | NVMe SSD 1TB |
| Python | 3.10.20 (conda 管理) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| conda 环境 | `f5-tts`（独立环境，路径 `~/.conda/envs/f5-tts`） |

### 与 Index-TTS 环境的对比

| 组件 | Index-TTS (`index-tts`) | F5-TTS (`f5-tts`) |
|------|------------------------|-------------------|
| Python | 3.10.20 | 3.10.20 |
| PyTorch | 2.10.0+cu128 | 2.10.0+cu128 |
| CUDA | 12.8 | 12.8 |
| 包管理 | UV (pyproject.toml) | pip (pyproject.toml) |
| Gradio | 5.45.0 | 6.11.0 |
| 入口 | `webui.py` | `src/f5_tts/infer/infer_gradio.py` |
| 默认端口 | 7860 | 7860 |

> 两个项目使用独立的 conda 环境，避免依赖版本冲突。

---

## 2. 部署步骤

### 步骤 1：克隆源码

```bash
cd d:\Workspace\MyAIModels\F5-TTS
git clone git@github.com:SWivid/F5-TTS.git .
```

> 如果目标目录非空（已有文件），先克隆到临时目录再移入：
> ```bash
> git clone git@github.com:SWivid/F5-TTS.git d:/path/to/temp
> cp -r d:/path/to/temp/* d:/Workspace/MyAIModels/F5-TTS/
> cp -r d:/path/to/temp/.git d:/Workspace/MyAIModels/F5-TTS/
> rm -rf d:/path/to/temp
> ```

### 步骤 2：创建 Conda 虚拟环境

```bash
conda create -n f5-tts python=3.10 -y
conda activate f5-tts
```

### 步骤 3：安装 PyTorch（从已有环境拷贝 + 补依赖）

如果本机已有同版本 PyTorch 的 conda 环境（如 `index-tts`），可以拷贝避免重新下载 ~2.5GB：

```bash
# 源环境路径和目标环境路径，按实际情况修改
SRC="C:/Users/hefen/miniconda3/envs/index-tts/lib/site-packages"
DST="C:/Users/hefen/.conda/envs/f5-tts/lib/site-packages"

# 拷贝核心包
cp -r "$SRC/torch" "$DST/"
cp -r "$SRC/torch-2.10.0+cu128.dist-info" "$DST/"
cp -r "$SRC/torchaudio" "$DST/"
cp -r "$SRC/torchaudio-2.10.0+cu128.dist-info" "$DST/"
cp -r "$SRC/torchgen" "$DST/"
```

然后补装 PyTorch 的 Python 依赖：

```bash
pip install typing_extensions sympy filelock networkx jinja2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **重要**：numpy 不能直接拷贝！拷贝后缺少 DLL 会报 `ImportError: DLL load failed`。
> 必须用 pip 重新安装：
> ```bash
> pip install numpy==1.26.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

验证 PyTorch：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 应输出: True NVIDIA GeForce RTX 5060 Laptop GPU
```

### 步骤 4：安装 F5-TTS 依赖

```bash
cd d:\Workspace\MyAIModels\F5-TTS
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 使用清华 PyPI 镜像加速。`-e` 为可编辑模式，修改源码后无需重新安装。

### 步骤 5：下载预训练模型

**TTS 模型**（从魔塔社区下载）：

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

下载后的目录结构：

```
ckpts/
├── F5TTS_v1_Base/
│   ├── model_1250000.safetensors   (~1.26 GB)
│   ├── vocab.txt
│   ├── configuration.json
│   └── README.md
├── vocos-mel-24khz/
│   ├── config.yaml
│   └── pytorch_model.bin
└── README.md
```

### 步骤 6：配置 HuggingFace 镜像（可选，持久化）

```bash
mkdir -p ~/.conda/envs/f5-tts/etc/conda/activate.d
echo 'export HF_ENDPOINT=https://hf-mirror.com' > ~/.conda/envs/f5-tts/etc/conda/activate.d/env_vars.sh
```

> 每次激活 `f5-tts` 环境时自动生效。

### 步骤 7：启动 Gradio Web UI

```bash
conda activate f5-tts
cd d:\Workspace\MyAIModels\F5-TTS
python -m f5_tts.infer.infer_gradio
```

或使用 CLI 入口（已安装为可编辑包后可用）：

```bash
f5-tts_infer-gradio
```

启动后浏览器访问 **http://127.0.0.1:7860**。

---

## 3. 踩坑记录与解决方案

### 坑 1：git clone 目标目录非空

**问题**：`git clone git@github.com:SWivid/F5-TTS.git .` 报错 `fatal: destination path '.' already exists and is not an empty directory.`

**原因**：目录中已有 `F5-TTS-部署计划.md` 文件。

**解决**：克隆到临时目录，然后 `cp -r` 移入，包括隐藏的 `.git` 目录。

### 坑 2：numpy 跨环境拷贝失败

**问题**：从 `index-tts` 环境拷贝 numpy 到 `f5-tts` 环境后，`import numpy` 报 `ImportError: DLL load failed while importing _multiarray_umath`。

**原因**：numpy 包含编译好的 C 扩展（`.pyd` 文件），这些文件依赖特定的 C runtime DLL 路径，直接拷贝 `.py` 和 `.pyd` 文件不够，还缺少关联的 DLL。

**解决**：删除拷贝的 numpy，用 pip 重新安装：
```bash
rm -rf site-packages/numpy* && pip install numpy==1.26.2
```

**教训**：**numpy 等含 C 扩展的包不要跨环境拷贝**，用 pip 安装更可靠。

### 坑 3：conda run 不支持多行 Python 脚本

**问题**：`conda run -n f5-tts python -c "import torch\nimport numpy\n..."` 报错 `NotImplementedError: Support for scripts where arguments contain newlines not implemented.`

**原因**：Windows 上的 conda run 不支持参数中包含换行符。

**解决**：
- 单行验证：`conda run -n f5-tts python -c "import torch; print(torch.__version__)"`
- 启动服务：直接用环境中的 Python 解释器，如 `"C:/Users/hefen/.conda/envs/f5-tts/python.exe" -u -m f5_tts.infer.infer_gradio`

### 坑 4：模型从 HuggingFace 下载极慢

**问题**：首次启动时，F5-TTS 自动从 HuggingFace 下载模型，国内网络速度接近 0。

**解决**：
- TTS 主模型：从魔塔社区（ModelScope）下载 `SWivid/F5-TTS_Emilia-ZH-EN`
- Vocos 声码器：魔塔上没有此模型，使用 `HF_ENDPOINT=https://hf-mirror.com` 从 HuggingFace 镜像下载

**教训**：**永远假设国内直连 HuggingFace 不可用**，提前从魔塔或 hf-mirror 下载好模型到本地。

### 坑 5：代码默认从 HuggingFace 加载模型

**问题**：即使模型已下载到本地，代码中的 `cached_path("hf://SWivid/F5-TTS/...")` 仍会联网验证。

**解决**：修改 `src/f5_tts/infer/infer_gradio.py`，将 `hf://` 远程路径替换为本地路径，并设置 `load_vocoder(is_local=True)`。详见 [第 4 节](#4-已做的代码修改)。

---

## 4. 已做的代码修改

以下修改让 F5-TTS 从本地 `ckpts/` 目录加载模型，避免联网：

**文件**：`src/f5_tts/infer/infer_gradio.py`

```python
# === 修改前（原始代码）===
DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

vocoder = load_vocoder()

def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

# === 修改后（本地路径）===
_DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "ckpts", "F5TTS_v1_Base")

DEFAULT_TTS_MODEL_CFG = [
    os.path.join(_DEFAULT_CKPT_DIR, "model_1250000.safetensors"),
    os.path.join(_DEFAULT_CKPT_DIR, "vocab.txt"),
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

_VOCOS_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "ckpts", "vocos-mel-24khz")
vocoder = load_vocoder(is_local=True, local_path=_VOCOS_LOCAL_DIR)

def load_f5tts():
    ckpt_path = DEFAULT_TTS_MODEL_CFG[0]
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)
```

> **注意**：这些修改意味着 `git pull` 更新源码时可能产生冲突。建议在 `git pull` 前用 `git stash` 暂存修改。

---

## 5. 日常使用

### 启动服务

```bash
conda activate f5-tts
cd d:\Workspace\MyAIModels\F5-TTS
python -m f5_tts.infer.infer_gradio
# 浏览器打开 http://127.0.0.1:7860
```

### CLI 推理

```bash
conda activate f5-tts
f5-tts_infer-cli --ref_audio "参考音频.wav" --ref_text "参考音频的文字内容" --gen_text "要合成的文本" --output "输出.wav"
```

### Python API 调用

```python
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
from f5_tts.model import DiT

# 加载模型
model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
ckpt_path = "ckpts/F5TTS_v1_Base/model_1250000.safetensors"
vocab_path = "ckpts/F5TTS_v1_Base/vocab.txt"
vocoder = load_vocoder(is_local=True, local_path="ckpts/vocos-mel-24khz")
model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)

# 推理
wav, sr = infer_process(ref_audio, ref_text, gen_text, model, vocoder)
```

---

## 6. 常见问题

### Q: pip install 很慢怎么办？

使用清华 PyPI 镜像：
```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或永久配置：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 模型下载不下来怎么办？

优先级：魔塔社区 > hf-mirror > HuggingFace 官方。

```bash
# 魔塔社区
python -c "from modelscope import snapshot_download; snapshot_download('SWivid/F5-TTS_Emilia-ZH-EN', local_dir='ckpts/F5TTS_v1_Base')"

# HuggingFace 镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download SWivid/F5-TTS --local-dir ./ckpts
```

### Q: CUDA out of memory 怎么办？

- 8GB 显存通常足够生成 30 秒以内的音频
- 超过 30 秒的文本建议分段生成
- 确保没有其他 GPU 占用程序（如 Index-TTS 同时运行）

### Q: git pull 后本地修改被覆盖？

```bash
git stash          # 暂存本地修改
git pull           # 拉取更新
git stash pop      # 恢复本地修改
```

### Q: conda 环境损坏了怎么办？

删掉重建即可，依赖可以重新安装：
```bash
conda remove -n f5-tts --all
conda create -n f5-tts python=3.10 -y
# 重新执行步骤 3-5
```

本地模型文件在 `ckpts/` 目录中不受影响。

---

## 附录：国内镜像源速查

| 用途 | 镜像地址 |
|------|---------|
| PyPI (pip) | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| HuggingFace | `https://hf-mirror.com` |
| 魔塔社区 (ModelScope) | `https://www.modelscope.cn` |
| Conda (conda-forge) | `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` |
| PyTorch 官方 (有墙) | `https://download.pytorch.org/whl/cu128` |
