# F5-TTS HTTP 微服务 + StoryTeller 集成

## Context

StoryTeller（快影工坊）当前使用 Edge-TTS 做配音。用户希望接入 F5-TTS 实现声音克隆。
两个项目因 numpy 版本冲突（StoryTeller <1.24 vs F5-TTS >=1.24）不能共存一个 Python 环境，
因此将 F5-TTS 封装为独立 HTTP 微服务，StoryTeller 通过 HTTP 调用。

## 架构

```
StoryTeller (numpy<1.24)
  |  httpx async HTTP
  v
F5-TTS HTTP Service (numpy 1.26, PyTorch, CUDA)
  ├── FastAPI + uvicorn
  ├── F5-TTS 模型 (启动时加载一次)
  ├── Whisper ASR (词级时间戳 → VTT 字幕)
  └── voices/ 目录 (预设多个音色)
```

## 阶段一：F5-TTS HTTP 服务（本项目的核心交付）

### 新建文件

#### 1. `src/f5_tts/service/__init__.py`
空文件，让 service 成为 Python 包。

#### 2. `src/f5_tts/service/voice_library.py`
音色预设管理。扫描 `voices/` 目录，每个音色由 `{name}.wav` + `{name}.txt` 组成。
```python
@dataclass
class VoicePreset:
    voice_id: str
    ref_audio_path: str
    ref_text: str
    description: str = ""

def scan_voices_dir(voices_dir: str) -> Dict[str, VoicePreset]
```

#### 3. `src/f5_tts/service/whisper_timestamps.py`
新建函数，复用已有的 `asr_pipe`，但开启 `return_timestamps="word"`。
不修改现有 `transcribe()` 函数。
```python
def transcribe_with_word_timestamps(audio_path, text, language=None) -> List[Tuple[float, float, str]]
# 返回: [(start, end, word), ...]
```

#### 4. `src/f5_tts/service/vtt_generator.py`
将词级时间戳转为 WebVTT 格式，按标点/时长分组为字幕条目。
```python
def generate_vtt(word_timestamps, output_path, max_chars_per_cue=40, max_duration=4.0) -> str
```

#### 5. `src/f5_tts/service/server.py` — 主文件
FastAPI 应用，启动时加载 F5-TTS 模型 + Vocos + Whisper。

**API 端点：**

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/voices` | 列出可用音色 |
| POST | `/tts` | 生成音频（返回 wav 路径 + 时长） |
| POST | `/tts/with-subtitles` | 生成音频 + VTT 字幕 |

**请求/响应示例：**
```
POST /tts/with-subtitles
Body: {"text": "你好世界", "voice_id": "default", "speed": 1.0}
Response: {"audio_path": "D:/.../tts_xxxx.wav", "vtt_path": "D:/.../tts_xxxx.vtt", "duration": 3.5}
```

**配置**（环境变量，均有默认值）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| F5TTS_HOST | 0.0.0.0 | 监听地址 |
| F5TTS_PORT | 8880 | 监听端口 |
| F5TTS_OUTPUT_DIR | output/tts_service | 输出目录 |
| F5TTS_VOICES_DIR | voices | 音色目录 |

**并发**：单线程 ThreadPoolExecutor，GPU 推理串行，FastAPI 处理异步 HTTP 层。

#### 6. `src/f5_tts/service/run_server.py`
启动入口。`python -m f5_tts.service.run_server`。

### 新建目录

#### `voices/`
音色预设目录。每个音色放两个同名文件：
```
voices/
├── default.wav    # 参考音频 (3-12秒)
└── default.txt    # 参考文本 (第一行是文字)
```
启动时把 `src/f5_tts/infer/examples/basic/basic_ref_zh.wav` 复制为 default 音色。

### 修改文件

#### `pyproject.toml`
新增可选依赖和入口：
```toml
[project.optional-dependencies]
service = ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"]

[project.scripts]
"f5-tts_service" = "f5_tts.service.run_server:main"
```

## 阶段二：StoryTeller 客户端（F5-TTS 项目做完后）

### 修改文件

#### `src/services/tts_service.py`
新增 `F5TTSService(TTSService)` 类：
- 用 httpx 异步调用 F5-TTS HTTP 服务
- 实现 `text_to_speech()` 和 `text_to_speech_with_subtitles()`
- 从服务端返回路径拷贝 wav/vtt 到本地输出目录

#### `src/video_narrator.py`
- `_create_tts_service()` 增加 F5-TTS 分支
- `get_supported_voices()` 根据 engine 配置返回对应音色列表

#### `config/settings.yaml`
新增 `tts.engine: "edge-tts"` 配置项和 F5-TTS 音色列表。

#### `requirements.txt`
新增 `httpx>=0.25.0`。

## 验证步骤

### 阶段一验证（F5-TTS 服务独立测试）

```bash
# 1. 启动服务
conda activate f5-tts
cd d:\Workspace\MyAIModels\F5-TTS
pip install fastapi "uvicorn[standard]"
python -m f5_tts.service.run_server

# 2. 另一个终端测试
curl http://localhost:8880/health
curl http://localhost:8880/voices
curl -X POST http://localhost:8880/tts/with-subtitles -H "Content-Type: application/json" -d "{\"text\": \"你好世界\", \"voice_id\": \"default\"}"
# 检查返回的 wav/vtt 文件是否有效
```

### 阶段二验证（StoryTeller 集成测试）

```python
# Python 快速测试
import asyncio
from src.services.tts_service import F5TTSService
async def test():
    svc = F5TTSService(voice="default")
    audio, vtt, dur = await svc.text_to_speech_with_subtitles("测试文本", "output/test")
    print(f"Audio: {audio}, VTT: {vtt}, Duration: {dur}s")
asyncio.run(test())
```
