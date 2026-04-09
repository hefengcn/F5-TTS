# 参考音频文件说明

本目录包含用于 F5-TTS 声音克隆实验的参考音频样本。

所有文件已统一转换为 **24kHz 单声道 16-bit PCM WAV** 格式，可直接用于推理。

## 文件统计

| 分类 | 数量 | 来源 | 许可 |
|------|------|------|------|
| 英文男声 | 5 | LibriSpeech dev-clean | CC BY 4.0 |
| 英文女声 | 5 | LibriSpeech dev-clean | CC BY 4.0 |
| 中文男声 | 15 | AISHELL-3 | Apache 2.0 |
| 中文女声 | 15 | AISHELL-3 | Apache 2.0 |
| **总计** | **40** | | |

## 英文样本 (LibriSpeech)

| 文件 | 时长 | 说话人 |
|------|------|--------|
| `en_male_01_librispeech_spk251.wav` | 9.8s | spk251 |
| `en_male_02_librispeech_spk2277.wav` | 6.6s | spk2277 |
| `en_male_03_librispeech_spk2412.wav` | 9.9s | spk2412 |
| `en_male_04_librispeech_spk2428.wav` | 6.8s | spk2428 |
| `en_male_05_librispeech_spk3752.wav` | 8.8s | spk3752 |
| `en_female_01_librispeech_spk1919.wav` | 11.1s | spk1919 |
| `en_female_02_librispeech_spk2035.wav` | 9.0s | spk2035 |
| `en_female_03_librispeech_spk3000.wav` | 7.2s | spk3000 |
| `en_female_04_librispeech_spk3081.wav` | 10.5s | spk3081 |
| `en_female_05_librispeech_spk3536.wav` | 10.8s | spk3536 |

## 中文样本 (AISHELL-3)

30 个不同说话人（5 男 5 女 × 3），44.1kHz 录音棚级原始录音。时长 3.0s ~ 8.4s。

## 使用方法

```bash
# CLI
f5-tts_infer-cli \
  --ref_audio src/f5_tts/infer/examples/ref_audio/en_female_01_librispeech_spk1919.wav \
  --gen_text "This is a voice cloning test using LibriSpeech reference audio."

# Python API
from f5_tts.api import F5TTS
f5tts = F5TTS()
wav, sr, spec = f5tts.infer(
    ref_file="src/f5_tts/infer/examples/ref_audio/zh_male_03_aishell3_SSB0394.wav",
    gen_text="你好，这是中文语音克隆测试。"
)
```

## 注意事项

- 参考音频建议 3~12 秒，所有文件已在范围内
- `ref_text` 可留空，系统自动调用 Whisper 转录
- 全部为合法开源数据集，可自由使用
