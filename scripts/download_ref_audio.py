"""
从 AISHELL-3 (hf-mirror.com) 批量下载参考音频
- 筛选 3~12 秒时长的文件
- 按说话人性别分类，排除已使用的说话人
- 输出 24kHz 单声道 WAV
"""
import subprocess, os, sys, json, tempfile

MIRROR = "https://hf-mirror.com"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "f5_tts", "infer", "examples", "ref_audio")
OUT_DIR = os.path.abspath(OUT_DIR)

# 已使用的说话人
USED = {"SSB0005","SSB0009","SSB0011","SSB0016","SSB0018",
        "SSB0033","SSB0038","SSB0057","SSB0073","SSB0149"}

TARGET_MALE = 15
TARGET_FEMALE = 15

def run(cmd, timeout=30):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def get_speaker_info():
    """下载并解析 spk-info.txt"""
    url = f"{MIRROR}/datasets/AISHELL/AISHELL-3/resolve/main/spk-info.txt"
    tmp = os.path.join(tempfile.gettempdir(), "spk_info.txt")
    if not os.path.exists(tmp):
        run(["curl", "-sL", "--max-time", "30", "-o", tmp, url], timeout=40)

    males, females = [], []
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                spk, _, gender = parts[0], parts[1], parts[2]
                if spk in USED:
                    continue
                if gender == "male":
                    males.append(spk)
                elif gender == "female":
                    females.append(spk)
    return males, females

def list_train_files(spk, limit=50):
    """列出说话人的 train 集文件"""
    url = f"{MIRROR}/api/datasets/AISHELL/AISHELL-3/tree/main/train/wav/{spk}?limit={limit}"
    out = run(["curl", "-sL", "--max-time", "20", url], timeout=30)
    try:
        data = json.loads(out)
        return [d["path"].split("/")[-1] for d in data]
    except:
        return []

def download_and_check(spk, fname):
    """下载单个文件并检查时长，返回 (临时路径, 时长) 或 None"""
    url = f"{MIRROR}/datasets/AISHELL/AISHELL-3/resolve/main/train/wav/{spk}/{fname}"
    tmp = os.path.join(tempfile.gettempdir(), f"ref_dl_{spk}_{fname}")
    run(["curl", "-sL", "--max-time", "30", "-o", tmp, url], timeout=40)

    # 验证是否为有效 WAV
    ftype = run(["file", tmp])
    if "RIFF" not in ftype and "WAVE" not in ftype:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    dur_str = run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                   "-of", "csv=p=0", tmp])
    try:
        dur = float(dur_str)
    except:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

    return tmp, dur

def convert_to_24k(src, dst):
    """转换为 24kHz 单声道 WAV"""
    run(["ffmpeg", "-y", "-i", src, "-ar", "24000", "-ac", "1", "-t", "12", dst],
        timeout=30)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("获取说话人信息...")
    males, females = get_speaker_info()
    print(f"可用说话人: {len(males)} 男, {len(females)} 女")

    # 选择前 N 个说话人（打乱以增加多样性）
    import random
    random.seed(42)
    random.shuffle(males)
    random.shuffle(females)

    male_speakers = males[:TARGET_MALE * 2]  # 多选一些，留余量
    female_speakers = females[:TARGET_FEMALE * 2]

    results = {"male": [], "female": []}

    for gender, speakers, target in [
        ("male", male_speakers, TARGET_MALE),
        ("female", female_speakers, TARGET_FEMALE),
    ]:
        print(f"\n=== 下载 {gender} 声音 ({target} 个) ===")
        count = 0

        for spk in speakers:
            if count >= target:
                break

            # 获取文件列表
            files = list_train_files(spk)
            if not files:
                print(f"  {spk}: 无文件，跳过")
                continue

            # 按文件名排序，优先尝试编号较大的（通常更长）
            files.sort(reverse=True)

            best = None
            for fname in files[:15]:  # 最多试 15 个文件
                result = download_and_check(spk, fname)
                if result is None:
                    continue
                tmp, dur = result
                if 4 <= dur <= 12:
                    best = (tmp, dur, fname)
                    break  # 找到合适长度的
                elif 3 <= dur < 4 and best is None:
                    best = (tmp, dur, fname)
                else:
                    if os.path.exists(tmp):
                        os.remove(tmp)

            if best is None:
                print(f"  {spk}: 无合适文件")
                continue

            tmp, dur, fname = best
            count += 1
            out_name = f"zh_{gender}_{count:02d}_aishell3_{spk}.wav"
            out_path = os.path.join(OUT_DIR, out_name)

            convert_to_24k(tmp, out_path)
            if os.path.exists(tmp):
                os.remove(tmp)

            print(f"  [{count}/{target}] {out_name}  ({dur:.1f}s)  ← {fname}")
            results[gender].append((out_name, dur, spk))

    print(f"\n=== 完成 ===")
    print(f"男声: {len(results['male'])} 个")
    print(f"女声: {len(results['female'])} 个")
    print(f"总计: {len(results['male']) + len(results['female'])} 个")
    print(f"输出目录: {OUT_DIR}")

if __name__ == "__main__":
    main()
