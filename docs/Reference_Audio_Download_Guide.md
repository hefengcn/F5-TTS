# 参考音频下载经验总结

**日期**: 2026-04-09
**背景**: 为 F5-TTS 声音克隆实验寻找高质量参考音频的完整踩坑记录

---

## 1. 法律红线

**不要使用名人声音！**

中国《民法典》第1023条明确保护**声音权**（参照肖像权）。未经本人同意，复制、使用他人声音属于侵权。相声、影视录音同时受著作权法保护。提交到 GitHub 公开仓库更构成传播行为。

**合法来源只有两类：**
- 开源数据集（Apache 2.0 / CC0 / CC-BY 等许可）
- 自己录制（需获得说话人书面授权）

---

## 2. 可用的开源语音数据集

### 2.1 中文首选：AISHELL-3

| 属性 | 值 |
|------|-----|
| 许可 | **Apache 2.0**（可商用） |
| 说话人 | 218 人（有性别、年龄、口音标注） |
| 音质 | **44.1kHz, 16-bit WAV**，录音棚级品质 |
| 时长 | 85 小时 |
| 用途 | 专为多说话人 TTS 设计 |
| 说话人信息 | `spk-info.txt` 含性别/年龄段/口音区 |

### 2.2 英文首选：LibriSpeech dev-clean

| 属性 | 值 |
|------|-----|
| 许可 | **CC BY 4.0** |
| 说话人 | ~40 人（validation 集），有 speaker_id |
| 音质 | 16kHz FLAC，清晰朗读 |
| 时长 | validation 约 5.4 小时 |
| 来源 | LibriVox 公共领域有声读物 |
| 特点 | 英文 TTS 基准测试标准数据集 |

### 2.3 备选数据集

| 数据集 | 语言 | 许可 | 音质 | 说话人 | 说明 |
|--------|------|------|------|--------|------|
| AISHELL-1 | 中文 | Apache 2.0 | 16kHz | 400人 | ASR 用，音质稍低 |
| THCHS-30 | 中文 | Apache 2.0 | 16kHz | 50人 | 清华大学，35小时 |
| Common Voice | 多语言 | **CC0** | 48kHz MP3 (参差) | 数千人 | 质量波动大，需筛选 |
| ST-CMDS | 中文 | CC BY-NC-ND | 16kHz | 855人 | **不可商用** |
| MAGICDATA | 中文 | CC BY-NC-ND | 16kHz | 1080人 | **不可商用** |
| VCTK | 英文 | CC BY 4.0 | 48kHz WAV | 109人 | 多口音，parquet 格式 |
| LJSpeech | 英文 | Public Domain | 22kHz | 1人(女) | 单说话人，不适合多声音克隆 |

---

## 3. 下载踩坑全记录

### 坑 1：HuggingFace 国内被墙

```
ConnectionError: Couldn't reach 'AISHELL/AISHELL-3' on the Hub
```

**解决**：使用国内镜像 `hf-mirror.com`

```python
# 方法 1：环境变量（Python 脚本开头设置）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 方法 2：命令行设置（注意：对 datasets 库可能不生效）
HF_ENDPOINT=https://hf-mirror.com python script.py
```

**注意**：`datasets` 库的 `load_dataset()` 流式下载有时不走 `HF_ENDPOINT`，此时需要直接用 REST API 下载单个文件。

### 坑 2：OpenSLR 下载链接被重定向

```bash
# 返回 HTML 而非文件（278 字节）
curl -sL -o dev.tar.gz "https://www.openslr.org/resources/93/dev.tar.gz"
file dev.tar.gz  # → HTML document
```

**原因**：OpenSLR 使用 JavaScript 重定向，`curl` 无法处理。

**解决**：通过 hf-mirror.com 的 REST API 直接下载单个文件（见第 4 节）。

### 坑 3：Wikimedia Commons URL 编码问题

```bash
# 含空格/特殊字符的文件名直接 curl 返回 HTML
curl -sL -o xxx.ogg "https://upload.wikimedia.org/.../From the Battlefields of France.ogg"
# → HTML document (404 页面)
```

**规律**：只有 URL 中**不含空格和特殊字符**的文件才能直接下载。

**结论**：Wikimedia 作为参考音频来源**不可靠**，成功率低，且大多是历史录音音质差。**不推荐使用。**

### 坑 4：ffmpeg 不能原地覆盖同名文件

```bash
ffmpeg -y -i input.wav -ar 24000 input.wav
# → FFmpeg cannot edit existing files in-place.
```

**解决**：先重命名为临时文件，转换后再删除。

### 坑 5：datasets 库依赖 lzma，Windows base 环境缺 DLL

```
ImportError: DLL load failed while importing _lzma
```

**解决**：使用项目的 conda 环境（`f5-tts`），该环境已正确安装所有依赖。

```bash
# 不要用 base 环境的 python
C:/Users/hefen/.conda/envs/f5-tts/python.exe script.py
```

### 坑 6：AISHELL-3 test 集每人只有 1 条，太短

test 集文件普遍 1-3 秒，不适合做参考音频。

**解决**：从 **train 集**下载，文件编号越大通常句子越长（5-8 秒的理想长度）。

### 坑 7：LibriSpeech 在 hf-mirror 上是 parquet 格式，没有原始 FLAC

hf-mirror 上的 `openslr/librispeech_asr` 将音频存储在 parquet 文件中（`clean/validation/0000.parquet`），而非原始 FLAC。不能用 curl 直接下载单个音频文件。

**解决**：下载 parquet → pyarrow 读取 → 提取音频 bytes → ffmpeg 转换。详见第 4 节方法 D。

### 坑 8：datasets 库解码 parquet 音频需要 torchcodec

```
ImportError: To support decoding audio data, please install 'torchcodec'.
```

安装 torchcodec 后又遇到 DLL 加载失败（Windows FFmpeg 版本不匹配）。

**结论**：**不要用 `datasets` 库的流式 API 处理 LibriSpeech**。直接用 pyarrow 读取 parquet 中的原始 bytes + ffmpeg 转换，更可靠。

### 坑 9：批量下载脚本中 bash `printf` 格式化问题

```bash
# 报错：value too great for base
echo "$(printf '%02d' $i)"
# 原因：bash 将 02d 当成八进制数字解析
```

**解决**：在 Python 脚本中处理编号逻辑，避免在 bash 循环中使用 printf 格式化。

---

## 4. 验证有效的下载方法

### 方法 A：hf-mirror REST API 直接下载 WAV（推荐，适用 AISHELL-3）

```bash
MIRROR="https://hf-mirror.com"

# 第 1 步：列出文件
curl -sL "${MIRROR}/api/datasets/AISHELL/AISHELL-3/tree/main/train/wav/SSB0005?limit=10"
# 返回 JSON，每个元素有 "path" 字段

# 第 2 步：下载单个文件
curl -sL -o SSB00050020.wav \
  "${MIRROR}/datasets/AISHELL/AISHELL-3/resolve/main/train/wav/SSB0005/SSB00050020.wav"

# 第 3 步：验证
file SSB00050020.wav
# → RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 44100 Hz ✓
```

### 方法 B：parquet 提取音频 bytes（推荐，适用 LibriSpeech）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import tempfile, subprocess

# 1. 下载 parquet 文件（hf-mirror 自动缓存，只下一次）
local_path = hf_hub_download(
    'openslr/librispeech_asr',
    'clean/validation/0000.parquet',
    repo_type='dataset',
    cache_dir=tempfile.gettempdir() + '/hf_cache'
)

# 2. 读取 parquet
table = pq.read_table(local_path)
rows = table.to_pydict()
# rows['audio'][i]['bytes'] → FLAC 原始字节
# rows['speaker_id'][i]     → 说话人 ID
# rows['text'][i]           → 转录文本

# 3. 提取并转换
audio_bytes = rows['audio'][0]['bytes']
tmp_flac = os.path.join(tempfile.gettempdir(), 'tmp.flac')
with open(tmp_flac, 'wb') as f:
    f.write(audio_bytes)

subprocess.run([
    'ffmpeg', '-y', '-i', tmp_flac,
    '-ar', '24000', '-ac', '1', '-t', '12', 'output.wav'
])
os.remove(tmp_flac)
```

**关键点**：
- `hf_hub_download` 走 `HF_ENDPOINT` 镜像，国内可访问
- parquet 中 `audio` 列是 `{'bytes': ..., 'path': ...}` 字典
- 音频 bytes 是 FLAC 格式，需 ffmpeg 转 WAV
- 整个 validation 集在一个 parquet 文件中（2703 条），一次下载全部可用

### 方法 C：获取说话人性别信息

```bash
# AISHELL-3 说话人信息
curl -sL "https://hf-mirror.com/datasets/AISHELL/AISHELL-3/resolve/main/spk-info.txt"
# 格式：speaker_id \t age_group \t gender \t accent
# 例：SSB0005  B  female  north

# LibriSpeech 说话人信息（已内置于代码中的映射表）
# LibriSpeech dev-clean 约 40 个说话人，性别需查 SPEAKER.TXT 或使用已知映射
```

### 方法 D：HuggingFace datasets 流式下载（不稳定，不推荐）

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
import soundfile as sf

ds = load_dataset('AISHELL/AISHELL-3', split='train', streaming=True)
for sample in ds:
    audio = sample['audio']
    sf.write('output.wav', audio['array'], audio['sampling_rate'])
    break
```

**注意**：方法 D 需要先读取数据集 schema（多次 API 调用），网络不稳定时容易超时。LibriSpeech 还需要 torchcodec 且 Windows 下有 DLL 兼容问题。**优先用方法 A 或 B。**

---

## 5. 音频处理标准流程

```bash
# 转为 F5-TTS 标准格式：24kHz 单声道 WAV
ffmpeg -y -i input.wav -ar 24000 -ac 1 -t 12 output.wav

# 参数说明：
#   -ar 24000  → 重采样到 24kHz
#   -ac 1      → 单声道
#   -t 12      → 截取前 12 秒（参考音频不宜过长）
```

### 参考音频筛选标准

| 条件 | 要求 | 原因 |
|------|------|------|
| 时长 | 4~12 秒最佳 | 太短特征不足，太长推理变慢 |
| 格式 | WAV 优先 | FLAC/OGG/MP3 也可，但需 ffmpeg 转换 |
| 采样率 | ≥16kHz 原始 | 重采样到 24kHz 不会引入额外失真 |
| 信噪比 | 安静环境录制 | 背景噪音会严重影响克隆质量 |
| 参考文本 | 已知更好 | 留空时自动 ASR 转录，但可能不准 |

---

## 6. 批量下载脚本

项目中提供了两个自动化脚本：

| 脚本 | 用途 | 使用方法 |
|------|------|----------|
| `scripts/download_ref_audio.py` | 从 AISHELL-3 批量下载中文参考音频 | 修改脚本中 `TARGET_MALE`/`TARGET_FEMALE` 数量后运行 |

运行方式：
```bash
C:/Users/hefen/.conda/envs/f5-tts/python.exe scripts/download_ref_audio.py
```

脚本特性：
- 通过 hf-mirror REST API 下载（不依赖 datasets 库）
- 自动解析 `spk-info.txt` 获取说话人性别
- 每个说话人尝试多个文件，筛选 4~12 秒的片段
- 自动转换为 24kHz 单声道 WAV
- 输出到 `src/f5_tts/infer/examples/ref_audio/`

---

## 7. 快速参考卡片

```
需要参考音频？
  │
  ├─ 中文 → AISHELL-3 (hf-mirror.com, Apache 2.0, 44.1kHz)
  │        API: /api/datasets/AISHELL/AISHELL-3/tree/main/{split}/wav/{speaker}
  │        DL:  /datasets/AISHELL/AISHELL-3/resolve/main/{path}
  │        性别: spk-info.txt
  │
  ├─ 英文 → LibriSpeech (hf-mirror.com, CC BY 4.0, 16kHz)
  │        方法: 下载 parquet → pyarrow 提取 bytes → ffmpeg 转 WAV
  │        路径: openslr/librispeech_asr / clean/validation/0000.parquet
  │        性别: SPEAKER.TXT 或内置映射表
  │
  ├─ 备选 → Common Voice (CC0, 多语言, 质量参差)
  │        https://commonvoice.mozilla.org/en/datasets
  │
  └─ 禁止 → 名人声音（违反《民法典》第1023条）
```

## 8. 当前参考音频库

`src/f5_tts/infer/examples/ref_audio/` 目录，共 40 个文件：

| 分类 | 数量 | 来源 | 时长范围 |
|------|------|------|----------|
| 英文男声 | 5 | LibriSpeech dev-clean | 6.6~9.9s |
| 英文女声 | 5 | LibriSpeech dev-clean | 7.2~11.1s |
| 中文男声 | 15 | AISHELL-3 train | 3.1~7.0s |
| 中文女声 | 15 | AISHELL-3 train | 3.0~8.4s |
