import os
import subprocess
import json
import asyncio
import edge_tts

# Slide narration scripts
slide_scripts = {
    1: (
        "大家好，今天我們要報告的主題是基於資料探勘標準流程，CRISP-DM，"
        "進行的 50 家新創公司獲利預測分析。在新創投資與營運中，資源配置效率直接決定公司成敗。"
        "我們將透過量化機器學習方法，分析各項財務指標對利潤的邊際推動力。"
    ),
    2: (
        "我們面臨的核心商業痛點，在於傳統預算分配往往依賴主觀經驗。"
        "因此，我們期望透過多元線性迴歸模型，將這些關係定量化，"
        "讓決策者清楚知道每一塊錢投入能回收多少利潤。"
    ),
    3: (
        "我們讀取數據後進行了基本的質量檢驗，確認數據完整。"
        "從皮爾森相關係數中可以發現，研發支出幾乎與利潤同步成長，這是第一個關鍵線索。"
        "相反，行政支出似乎與利潤無顯著線性關係。"
    ),
    4: (
        "在資料準備階段，我們對類別欄位進行了獨熱編碼。"
        "這裡有一個重要的統計學細節：為了避免虛擬變數陷阱，我們丟棄了 California 作為基準點，"
        "僅引入 Florida 與 New York 兩列。接著以 80/20 比例切割資料以備訓練。"
    ),
    5: (
        "我們比較了 5 種不同的特徵數組合在測試集上的預測能力。"
        "當我們把全部變數，包括行政開支與地理位置都塞進模型時，測試集 RMSE 卻由 8206 增加到 9055，"
        "調整後決定係數更由 0.89 跌至 0.77。這是一個非常典型的過擬合案例，"
        "說明簡化模型往往具備更好的泛化效果。"
    ),
    6: (
        "為了確保模型只保留最關鍵的因子，我們使用了五種不同的特徵選擇方法。"
        "不論是線性模型還是非線性模型，研發支出都穩居第一，行銷支出則緊追在後。"
        "這給予我們充足的依據，僅保留這兩個核心變數。"
    ),
    7: (
        "這是我們最終選定的雙特徵模型公式。"
        "截距項為 50,286 元，代表在不投入研發與行銷的情況下，企業的基底利潤。"
        "研發的係數為 0.81，表明其對獲利拉動的超高效率；行銷係數為 0.03。"
        "殘差圖檢定也證明了此模型的統計有效性。"
    ),
    8: (
        "最後，我們將模型部署為 Streamlit 網頁，供業務團隊動態估算預算回報率。"
        "基於我們的定量結論，有三大建議：首先是大幅優先投資研發；"
        "其次是避免盲目燒行銷預算；最後是極端精實行政開銷。感謝大家的聆聽，以上是我們的簡報。"
    )
}

async def generate_voiceover():
    workspace = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(workspace, "my_video", "assets", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    voice = "zh-TW-YunJheNeural"
    durations = {}
    
    # Path to ffprobe in env/ffmpeg/
    ffprobe_path = os.path.join(workspace, "env", "ffmpeg", "ffmpeg-8.1.1-essentials_build", "bin", "ffprobe.exe")
    if not os.path.exists(ffprobe_path):
        ffprobe_path = "ffprobe" # fallback
        
    for slide_id, text in slide_scripts.items():
        filename = f"slide{slide_id}.mp3"
        filepath = os.path.join(audio_dir, filename)
        
        print(f"Generating audio for Slide {slide_id}...")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filepath)
        
        # Get duration using ffprobe
        try:
            cmd = [
                ffprobe_path, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", filepath
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            duration = float(result.stdout.strip())
            # Add a small buffer of 0.8s for transitions and breathing room
            durations[slide_id] = duration + 0.8
            print(f"Slide {slide_id} duration: {duration:.2f}s (configured: {durations[slide_id]:.2f}s)")
        except Exception as e:
            print(f"Error checking duration for slide {slide_id}: {e}")
            durations[slide_id] = 15.0  # default fallback
            
    # Write durations to JSON
    with open(os.path.join(workspace, "my_video", "durations.json"), "w", encoding="utf-8") as f:
        json.dump(durations, f, indent=2, ensure_ascii=False)
        
    # Concatenate audio files using ffmpeg
    print("Concatenating audio files...")
    ffmpeg_path = os.path.join(workspace, "env", "ffmpeg", "ffmpeg-8.1.1-essentials_build", "bin", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg" # fallback
        
    concat_txt_path = os.path.join(audio_dir, "concat.txt")
    with open(concat_txt_path, "w", encoding="utf-8") as f:
        for slide_id in sorted(slide_scripts.keys()):
            f.write(f"file 'slide{slide_id}.mp3'\n")
            
    narration_path = os.path.join(workspace, "my_video", "assets", "narration.mp3")
    concat_cmd = [
        ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt_path,
        "-c", "copy", narration_path
    ]
    try:
        subprocess.run(concat_cmd, check=True)
        print(f"Master narration audio generated at {narration_path}")
    except Exception as e:
        print(f"Error concatenating audio: {e}")
        
    os.remove(concat_txt_path)
    print("Audio generation complete.")

if __name__ == "__main__":
    asyncio.run(generate_voiceover())
