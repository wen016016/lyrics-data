# -*- coding: utf-8 -*-
"""
add_song_new.py - 歌詞自動處理主流程

下載音檔 -> 時間軸對齊 (tools/aligner.py) -> 標振假名 -> 翻譯 -> 寫入 songs.json

時間軸對齊走全域 CTC forced alignment，start/end 都來自音訊 frame，
不用字數推算（細節見 aligner.py 的模組說明）。
語言由歌詞的文字系統自動判斷，日文以外的語言也支援。

翻譯會依現有的 API key 自動選 Claude 或 OpenAI；沒有 key 就跳過，
時間軸與振假名照跑，之後可用 --import-zh 補。

特殊念法用 `表記{讀音}` 標註，例如 365{エブリデイ}：
畫面顯示表記，對齊與振假名用讀音。

使用方式:
  python add_song_new.py --add --title "歌名" --artist "歌手" --video-id "影片ID" --lyrics lyrics.txt
  python add_song_new.py --import-zh "歌名" --zh-file translation.txt
  python add_song_new.py --process-pending
"""

import json, os, re, sys, argparse, wave
import time as time_module
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SONGS_JSON = os.path.join(SCRIPT_DIR, '..', 'songs.json')
TEMP_AUDIO = os.path.join(SCRIPT_DIR, 'temp_audio')

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Windows 主機的 console 預設是 cp950/cp1252，印中文會直接 UnicodeEncodeError
# （連 --help 都會炸）。這裡把 stdout/stderr 強制轉成 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 由 --translate-model 覆寫；None 表示用後端預設模型
TRANSLATE_MODEL = None


def strip_ruby(html_text):
    text = re.sub(r'<rt>.*?</rt>', '', html_text)
    text = re.sub(r'</?ruby>', '', text)
    return text.strip()

def fmt_time(seconds):
    return f"{int(seconds)//60}:{int(seconds)%60:02d}"

def check_dep(name):
    try: __import__(name); return True
    except ImportError: return False

# \u8b80\u97f3\u6a19\u8a3b\u8a9e\u6cd5: \u8868\u8a18{\u8b80\u97f3}
#   365{\u30a8\u30d6\u30ea\u30c7\u30a4}     -> \u756b\u9762\u986f\u793a 365\uff0c\u5c0d\u9f4a\u8207\u632f\u5047\u540d\u90fd\u7528\u300c\u30a8\u30d6\u30ea\u30c7\u30a4\u300d
#   mind{\u30de\u30a4\u30f3\u30c9}       -> \u65e5\u6587\u6b4c\u88e1\u7684\u82f1\u6587\u6bb5\u843d
#   \u55e4\u3046{\u308f\u3089\u3046}         -> \u6b4c\u624b\u7528\u4e86\u975e\u6a19\u6e96\u8b80\u97f3
#
# \u70ba\u4ec0\u9ebc\u9700\u8981\u9019\u500b\uff1a\u719f\u5b57\u8a13\u3001\u7fa9\u8a13\u3001\u6b4c\u624b\u81ea\u5275\u8b80\u97f3\u9019\u985e\u300c\u5b57\u9762\u63a8\u4e0d\u51fa\u8b80\u97f3\u300d\u7684\u60c5\u6cc1\uff0c
# \u4efb\u4f55\u81ea\u52d5\u8f49\u63db\u90fd\u731c\u4e0d\u5230\uff08pykakasi \u53ea\u6703\u628a 365 \u8b80\u6210 \u3055\u3093\u308d\u304f\u3054\uff09\u3002
# \u6b4c\u8a5e\u662f\u4f60\u63d0\u4f9b\u7684\u771f\u503c\uff0c\u8b80\u97f3\u4e5f\u4e00\u6a23 \u2014\u2014 \u8b93\u4f60\u76f4\u63a5\u6a19\uff0c\u6bd4\u8b93\u7a0b\u5f0f\u731c\u53ef\u9760\u3002
READING_ANNOT = re.compile(r'([^\s{}]+)\{([^{}]+)\}')


def parse_reading_annotations(raw):
    """\u628a\u4e00\u884c\u6b4c\u8a5e\u62c6\u6210 [(\u8868\u8a18, \u8b80\u97f3 or None), ...]\u3002"""
    segs = []
    pos = 0
    for m in READING_ANNOT.finditer(raw):
        if m.start() > pos:
            segs.append((raw[pos:m.start()], None))
        segs.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(raw):
        segs.append((raw[pos:], None))
    return segs or [(raw, None)]


def display_text(raw):
    """\u53bb\u6389\u8b80\u97f3\u6a19\u8a3b\uff0c\u5f97\u5230\u5be6\u969b\u8981\u986f\u793a\u7684\u6b4c\u8a5e\u3002"""
    return ''.join(s for s, _ in parse_reading_annotations(raw))


def alignment_text(raw):
    """\u628a\u6a19\u8a3b\u904e\u7684\u6bb5\u843d\u63db\u6210\u8b80\u97f3\uff0c\u5f97\u5230\u5c0d\u9f4a\u8981\u7528\u7684\u6587\u5b57\u3002"""
    return ''.join(r if r else s for s, r in parse_reading_annotations(raw))


def has_reading_annotations(raw):
    return bool(READING_ANNOT.search(raw))


# 合唱標記: 行首的 `+` 表示「這句和上一句同時唱」
#   我要飛翔
#   +和聲的另一句
#
# 為什麼要標而不是自動偵測：對齊的兩個機制（全域 CTC Viterbi 與 Whisper 錨點
# + 序列比對）都建立在「歌聲是單一條單調的序列」這個前提上，
# 結構上不可能吐出「同一時間有兩句」的結果。要偵測同時唱的兩個聲部，
# 得先把主唱和和聲分離成兩軌，那是另一個層級的問題。
# 你聽得出來哪幾句是疊在一起的，直接標最可靠。
JOIN_PREV = re.compile(r'^\s*\+\s*')


def parse_join_prev(raw):
    """回傳 (是否與上一句同時, 去掉標記後的歌詞)。"""
    m = JOIN_PREV.match(raw)
    return (True, raw[m.end():]) if m else (False, raw)


def text_to_furigana(raw):
    """\u6a19\u632f\u5047\u540d\u3002\u6709\u660e\u78ba\u8b80\u97f3\u6a19\u8a3b\u7684\u6bb5\u843d\u4e00\u5f8b\u63a1\u7528\u6a19\u8a3b\u503c\uff0c\u4e0d\u8b93 pykakasi \u8986\u84cb\u3002"""
    import pykakasi
    kks = pykakasi.kakasi()
    out = []
    for surface, reading in parse_reading_annotations(raw):
        if reading:
            out.append(f'<ruby>{surface}<rt>{reading}</rt></ruby>')
            continue
        for item in kks.convert(surface):
            orig, hira = item['orig'], item['hira']
            if orig != hira and any('\u4e00' <= c <= '\u9fff' for c in orig):
                out.append(f'<ruby>{orig}<rt>{hira}</rt></ruby>')
            else:
                out.append(orig)
    return ''.join(out)

def download_audio(video_id):
    import yt_dlp
    os.makedirs(TEMP_AUDIO, exist_ok=True)
    output_path = os.path.join(TEMP_AUDIO, f"{video_id}.wav")
    # Always re-download to avoid stale/corrupt files
    for ext in ['wav', 'webm', 'mp4', 'm4a', 'mp3', 'opus']:
        p = os.path.join(TEMP_AUDIO, f"{video_id}.{ext}")
        if os.path.exists(p):
            os.remove(p)

    print(f"  [audio] downloading...", flush=True)
    raw_path = os.path.join(TEMP_AUDIO, f"{video_id}.%(ext)s")

    # Try with FFmpeg postprocessor first (produces WAV directly)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": raw_path,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        if os.path.exists(output_path):
            _verify_wav(output_path)
            return output_path
    except Exception:
        pass  # FFmpeg not available, try without postprocessor

    # Download raw audio without FFmpeg
    print(f"  [audio] FFmpeg not found, downloading raw...", flush=True)
    ydl_opts2 = {
        "format": "bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": raw_path,
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts2) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    except Exception as e:
        print(f"  [audio] download failed: {e}")
        print(f"  Please manually save WAV to: {os.path.abspath(output_path)}")
        raise SystemExit(1)

    # Find downloaded file and convert to WAV using Python
    raw_file = None
    for ext in ['webm', 'mp4', 'm4a', 'opus', 'mp3']:
        p = os.path.join(TEMP_AUDIO, f"{video_id}.{ext}")
        if os.path.exists(p):
            raw_file = p
            break

    if not raw_file:
        print("  [audio] no audio file found after download")
        raise SystemExit(1)

    print(f"  [audio] converting {os.path.basename(raw_file)} -> wav ...", flush=True)
    _convert_to_wav(raw_file, output_path)

    # Clean up raw file
    if os.path.exists(raw_file):
        os.remove(raw_file)

    _verify_wav(output_path)
    return output_path


def _convert_to_wav(input_path, output_path):
    """Convert audio to 16kHz mono WAV using torch/torchaudio (no FFmpeg needed)"""
    import torch, torchaudio
    try:
        waveform, sr = torchaudio.load(input_path)
    except Exception as e:
        print(f"  [audio] torchaudio.load failed: {e}")
        print(f"  Please install FFmpeg: winget install Gyan.FFmpeg.Shared")
        raise SystemExit(1)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to 16kHz
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    torchaudio.save(output_path, waveform, 16000)


def _verify_wav(path):
    """Verify WAV file is valid and print info"""
    with wave.open(path, 'rb') as wf:
        dur = wf.getnframes() / wf.getframerate()
        ch = wf.getnchannels()
        sr = wf.getframerate()
        print(f"  [audio] OK: {dur:.1f}s, {ch}ch, {sr}Hz", flush=True)
    if dur < 10:
        print(f"  [audio] WARNING: audio too short ({dur:.1f}s), might be corrupt")
        raise SystemExit(1)

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        import torch, torchaudio
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 16000)
        audio = t.squeeze(0).numpy()
    return audio

def align_lyrics(audio, lyric_plains, align_texts=None, join_prev=None,
                 device="cpu", lang=None, separate=True, align_model=None,
                 anchor=True, whisper_model="large-v3"):
    """對齊歌詞到音訊。

    全域 CTC forced alignment (tools/aligner.py)：
    整首歌的所有歌詞串成一條 token 序列，對整首歌的 frame 跑一次 Viterbi，
    取全域最佳路徑。start / end 都來自真實音訊 frame。

    align_texts: 對齊專用文字（把 `365{エブリデイ}` 這種特殊念法換成讀音後的結果）。
                 None 時直接用 lyric_plains。

    這裡刻意沒有備援路徑。舊的「Whisper 轉錄 + 假名模糊比對」已經移除：
    它的比對門檻低到幾乎任何位置都能匹配，而且行尾時間是用字數推算的，
    產出的東西看起來有時間軸、實際上是錯的。寧可明確失敗，也不要靜靜寫入錯的資料。
    """
    audio_len = len(audio) / 16000.0
    import aligner
    results, used_lang = aligner.align_lines(
        audio, lyric_plains, align_texts=align_texts, join_prev=join_prev,
        lang=lang, device=device, separate=separate, model_name=align_model,
        anchor=anchor, whisper_model=whisper_model)
    aligner.report(results, lyric_plains)

    ok, why = _global_align_ok(results, audio_len)
    warning = None
    if not ok:
        warning = why
        print(f"  [align] 警告: 對齊結果沒通過合理性檢查: {why}", flush=True)
        print("  [align] 時間軸仍會寫入，請檢查上面標記的行", flush=True)
    else:
        print(f"  [align] 全域 CTC 對齊完成 ({used_lang}), "
              f"{len(results)}/{len(lyric_plains)} 行", flush=True)
    return [{'time': r['time'], 'end': r['end']} for r in results], warning


def _global_align_ok(results, audio_len):
    """全域對齊的合理性檢查。抓的是「整體錯位」，不是逐行精度。

    來源可信度分三級：
      ctc / anchor -- 邊界直接來自聲學對齊或 Whisper 錨點
      moved        -- 錨點落在間奏，改放到前後行之間真的有人聲的區段。
                      落點仍由人聲活動決定，算有音訊依據。
      interp       -- 完全沒有落點，前後線性插值，沒有任何音訊依據。
    """
    if not results:
        return False, "沒有結果"
    n = len(results)
    counts = {}
    for r in results:
        src = r.get('source') or 'interp'
        counts[src] = counts.get(src, 0) + 1
    breakdown = ', '.join(f"{k}={v}" for k, v in sorted(counts.items()))

    grounded = counts.get('ctc', 0) + counts.get('anchor', 0) + counts.get('moved', 0)
    if grounded < n * 0.7:
        return False, f"只有 {grounded}/{n} 行對到音訊 ({breakdown})"
    if n >= 8 and results[-1]['time'] < audio_len * 0.35:
        return False, "最後一行落點過早，時間軸疑似被壓縮"
    return True, ""


# ------------------------------------------------------------------ 翻譯

TRANSLATE_SYSTEM = """你是歌詞譯者，把歌詞翻成臺灣繁體中文。

規則：
1. 一行對一行。輸入 N 行就輸出 N 行，絕對不合併、不拆分、不增行、不減行。
2. 不要機翻。譯文要讀得像中文歌詞，語序自然，不保留日文語法殘留
   （例如不要出現「的事」「之物」這種直譯痕跡）。
3. 不要超譯。原文沒有的意象、形容詞、因果關係、主詞都不要自己加。
   原文含糊的地方，譯文就保持含糊，不要替作者解釋。
4. 原文的英文、專有名詞、擬聲詞按原樣保留，不要硬翻。
5. 反覆出現的同一句歌詞，翻譯要一致。
6. 空行或只有符號的行，原樣輸出。
7. 原文中的日文漢字「叶」（如「叶える」「叶う」，意思是實現/如願）請翻成對應的
   中文詞彙（例如「實現」「如願」「達成」），絕對不要把「叶」這個字原樣保留或
   轉寫成「葉」。這條規則只針對「叶」這個字，其他原本就是「葉」的漢字
   （例如「言葉」「葉」）不受影響，照常翻譯。

輸出格式：每行寫成「編號|譯文」，編號用輸入給的數字，中間用半角豎線分隔。
除此之外不要輸出任何說明、標題或空行。"""


def _chunk_indices(n, size):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _parse_numbered(text, lo, hi):
    """解析「編號|譯文」，回傳 {行號: 譯文}。"""
    out = {}
    for raw in text.split('\n'):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r'^\s*(\d+)\s*[|｜]\s*(.*)$', raw)
        if not m:
            m = re.match(r'^\s*(\d+)\s*[\.、)]\s*(.*)$', raw)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if lo <= idx < hi:
            out[idx] = m.group(2).strip()
    return out


def _translate_anthropic(numbered, lo, hi, model):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=16000,
        system=TRANSLATE_SYSTEM,
        messages=[{"role": "user", "content": numbered}],
    )
    if msg.stop_reason == "refusal":
        detail = getattr(msg, "stop_details", None)
        raise RuntimeError(f"模型拒絕翻譯這一段 ({getattr(detail, 'category', '?')})")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return _parse_numbered(text, lo, hi)


def _translate_openai(numbered, lo, hi, model):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": TRANSLATE_SYSTEM},
                  {"role": "user", "content": numbered}],
    )
    return _parse_numbered(resp.choices[0].message.content, lo, hi)


def pick_translate_backend():
    """依現有的 API key 選後端。回傳 (名稱, 函式, 預設模型) 或 None。"""
    if os.environ.get("ANTHROPIC_API_KEY") and check_dep("anthropic"):
        return ("Claude", _translate_anthropic, "claude-opus-5")
    if os.environ.get("OPENAI_API_KEY") and check_dep("openai"):
        return ("OpenAI", _translate_openai, "gpt-4o")
    return None


def translate_to_chinese(plains, model=None, chunk_size=25, verbose=True):
    """逐行翻譯成繁體中文。回傳與 plains 等長的 list，翻不到的位置是空字串。

    分段送出（預設 25 行）並用編號回填，而不是一次丟整首歌然後按行序讀回來 ——
    後者只要模型多一行少一行，整首歌的翻譯就會整體錯位一格，而且不會報錯。
    編號回填能明確知道哪幾行沒拿到譯文。
    """
    n = len(plains)
    backend = pick_translate_backend()
    if backend is None:
        print("  [translate] 沒有可用的 API key，跳過翻譯", flush=True)
        print("             設定 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 就會自動翻譯；", flush=True)
        print("             或先手動翻好，再用 --import-zh 匯入", flush=True)
        return [""] * n

    name, fn, default_model = backend
    model = model or default_model
    if verbose:
        print(f"  [translate] 使用 {name} ({model}) 翻譯 {n} 行...", flush=True)

    result = [""] * n
    for lo, hi in _chunk_indices(n, chunk_size):
        numbered = "\n".join(f"{i + 1}|{plains[i]}" for i in range(lo, hi))
        got = {}
        for attempt in (1, 2):
            try:
                got = fn(numbered, lo, hi, model)
            except Exception as e:
                print(f"  [translate] 第 {lo + 1}-{hi} 行失敗 ({e})", flush=True)
                got = {}
            missing = [i for i in range(lo, hi) if i not in got or not got[i]]
            if not missing or attempt == 2:
                break
            if verbose:
                print(f"  [translate] 第 {lo + 1}-{hi} 行缺 {len(missing)} 行，重試一次",
                      flush=True)
        for i, t in got.items():
            result[i] = t

    done = sum(1 for t in result if t)
    if done < n:
        missing = [i + 1 for i, t in enumerate(result) if not t]
        print(f"  [translate] 完成 {done}/{n} 行；缺這幾行需要人工補: {missing}",
              flush=True)
    elif verbose:
        print(f"  [translate] 完成 {n}/{n} 行", flush=True)
    return result


def process_song(song, do_translate=True, device="cpu", align_opts=None):
    title = song.get("title", "unknown")
    video_id = song.get("videoId")
    lines = song.get("lines", [])
    t0 = time_module.time()
    print(f"\n{'#'*60}\n  Processing: {title} ({len(lines)} lines)\n{'#'*60}", flush=True)

    audio_path = None
    if video_id:
        try: audio_path = download_audio(video_id)
        except SystemExit: raise
        except Exception as e: print(f"  Error: {e}")

    # `src` 保留帶讀音標註的原始歌詞，重跑時才不會因為 text 已被換成
    # furigana HTML 而丟掉標註。
    raw_all = [line.get("src") or strip_ruby(line["text"]) for line in lines]
    join_prev = []
    raw_texts = []
    for r in raw_all:
        j, body = parse_join_prev(r)
        join_prev.append(j)
        raw_texts.append(body)
    lyric_plains = [display_text(r) for r in raw_texts]
    align_texts = [alignment_text(r) for r in raw_texts]

    n_annot = sum(1 for r in raw_texts if has_reading_annotations(r))
    if n_annot:
        print(f"  讀音標註: {n_annot} 行採用你指定的讀音對齊", flush=True)
    n_join = sum(1 for j in join_prev if j)
    if n_join:
        print(f"  合唱標記: {n_join} 行與上一句同時亮", flush=True)

    try:
        if audio_path:
            audio = load_wav(audio_path)
            print(f"  Audio: {len(audio)/16000:.1f}s")
            aligned_lines, align_warning = align_lyrics(
                audio, lyric_plains,
                align_texts=align_texts,
                join_prev=join_prev, device=device,
                **(align_opts or {}))
            if align_warning:
                song["alignWarning"] = align_warning
            for i, a in enumerate(aligned_lines):
                if i >= len(lines):
                    break
                lines[i]["time"] = a["time"]
                lines[i]["end"] = a["end"]

        if check_dep("pykakasi"):
            for i, line in enumerate(lines):
                if '<ruby>' not in line["text"]:
                    line["text"] = text_to_furigana(raw_texts[i])
                line["plain"] = lyric_plains[i]
                # src 保留「原始標註」（含行首 + 與 {讀音}），重跑才不會遺失
                if has_reading_annotations(raw_texts[i]) or join_prev[i]:
                    line["src"] = raw_all[i]
                if "zh" not in line:
                    line["zh"] = ""
            print(f"  Furigana done: {len(lines)} lines")

        # 翻譯：只補還沒有譯文的行。整首一起送給模型（不是只送缺的那幾行），
        # 這樣重複出現的副歌譯法才會一致。
        if do_translate:
            missing = [i for i, line in enumerate(lines) if not line.get("zh", "")]
            if not missing:
                print("  翻譯已存在，跳過")
            else:
                zh = translate_to_chinese(lyric_plains, model=TRANSLATE_MODEL)
                for i in missing:
                    if i < len(zh) and zh[i]:
                        lines[i]["zh"] = zh[i]
    finally:
        # 放 finally：對齊或翻譯中途失敗時，暫存音檔一樣要刪掉，
        # 否則每失敗一次就留下數十 MB 的 wav。
        cleanup_temp_audio(audio_path)

    print(f"\n  Done! Time: {fmt_time(time_module.time()-t0)}", flush=True)
    return song


def cleanup_temp_audio(audio_path=None):
    """刪除暫存音檔。連同 temp_audio 內其他殘留一起清，目錄空了就移除。"""
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"  [cleanup] removed {audio_path}")
        except Exception:
            pass
    if not os.path.isdir(TEMP_AUDIO):
        return
    try:
        for fn in os.listdir(TEMP_AUDIO):
            p = os.path.join(TEMP_AUDIO, fn)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    print(f"  [cleanup] removed {p}")
                except Exception:
                    pass
        if not os.listdir(TEMP_AUDIO):
            os.rmdir(TEMP_AUDIO)
    except Exception:
        pass

def _read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def add_song(title, artist, video_id, lyrics_file, lyrics_by=None, music_by=None,
             color="#c9a96e", do_translate=True, device="cpu", align_opts=None,
             zh_file=None):
    if not os.path.exists(lyrics_file):
        print(f"  File not found: {lyrics_file}"); sys.exit(1)
    lyric_lines = _read_lines(lyrics_file)
    song = {"title": title, "artist": artist, "credits": {"lyrics": lyrics_by, "music": music_by},
            "color": color, "videoId": video_id, "lines": [{"time": 0, "text": l} for l in lyric_lines]}

    # 先給定的中文歌詞優先；沒給的行才會在 process_song 裡自動翻譯
    if zh_file:
        if not os.path.exists(zh_file):
            print(f"  File not found: {zh_file}"); sys.exit(1)
        zh_lines = [re.sub(r"^\d+[\.\)|｜]\s*", "", l) for l in _read_lines(zh_file)]
        if len(zh_lines) != len(lyric_lines):
            print(f"  中文歌詞 {len(zh_lines)} 行，日文歌詞 {len(lyric_lines)} 行，"
                  f"行數不一致，請先對齊行數"); sys.exit(1)
        for i, zh in enumerate(zh_lines):
            song["lines"][i]["zh"] = zh

    song = process_song(song, do_translate=do_translate, device=device, align_opts=align_opts)
    songs = json.load(open(SONGS_JSON, "r", encoding="utf-8")) if os.path.exists(SONGS_JSON) else []
    songs.append(song)
    with open(SONGS_JSON, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    print(f"  Done: '{title}' added to songs.json ({len(songs)} songs)")

def process_pending(device="cpu", do_translate=True, align_opts=None):
    pending_path = os.path.join(SCRIPT_DIR, 'pending_song.json')
    if not os.path.exists(pending_path):
        print("  pending_song.json not found"); sys.exit(1)
    with open(pending_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    song = {"title": data.get("title",""), "artist": data.get("artist",""),
            "credits": {"lyrics": data.get("credits",{}).get("lyrics"), "music": data.get("credits",{}).get("music")},
            "color": data.get("color","#c9a96e"), "videoId": data.get("videoId",""),
            "lines": [{"time": 0, "text": l} for l in data.get("lyrics",[])]}
    for i, zh in enumerate(data.get("zh", [])):
        if i < len(song["lines"]) and zh: song["lines"][i]["zh"] = zh
    song = process_song(song, do_translate=do_translate, device=device, align_opts=align_opts)
    songs = json.load(open(SONGS_JSON, "r", encoding="utf-8")) if os.path.exists(SONGS_JSON) else []
    songs.append(song)
    with open(SONGS_JSON, "w", encoding="utf-8") as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    print(f"  Done: '{song['title']}' added ({len(songs)} songs)")
    os.remove(pending_path)

def import_zh(title, zh_file):
    if not os.path.exists(zh_file):
        print(f"  File not found: {zh_file}"); sys.exit(1)
    with open(zh_file, "r", encoding="utf-8") as f:
        translations = [re.sub(r"^\d+[\.\)]\s*", "", l.strip()) for l in f if l.strip()]
    with open(SONGS_JSON, "r", encoding="utf-8") as f:
        songs = json.load(f)
    for song in songs:
        if song.get("title") == title:
            for i, zh in enumerate(translations):
                if i < len(song["lines"]): song["lines"][i]["zh"] = zh
            print(f"  Done: imported {len(translations)} lines to '{title}'")
            with open(SONGS_JSON, "w", encoding="utf-8") as f:
                json.dump(songs, f, ensure_ascii=False, indent=2)
            return
    print(f"  Song not found: {title}"); sys.exit(1)

def main():
    os.chdir(SCRIPT_DIR)
    p = argparse.ArgumentParser(description="Add Japanese song with auto timing + furigana")
    p.add_argument("--add", action="store_true")
    p.add_argument("--process-pending", action="store_true")
    p.add_argument("--import-zh", type=str)
    p.add_argument("--title", type=str)
    p.add_argument("--artist", type=str)
    p.add_argument("--video-id", type=str)
    p.add_argument("--lyrics", type=str)
    p.add_argument("--lyrics-by", type=str)
    p.add_argument("--music-by", type=str)
    p.add_argument("--color", type=str, default="#c9a96e")
    p.add_argument("--zh-file", type=str)
    p.add_argument("--no-translate", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lang", type=str, default=None,
                   help="歌詞語言碼 (ja/en/zh/ko/...)。預設從歌詞的文字系統自動判斷")
    p.add_argument("--align-model", type=str, default=None,
                   help="指定 HuggingFace CTC 對齊模型，覆蓋語言預設")
    p.add_argument("--no-separate", action="store_true",
                   help="不做人聲分離（快一些，但有合音/厚伴奏的歌會比較不準）")
    p.add_argument("--no-anchor", action="store_true",
                   help="不跑 Whisper 錨點（快很多，但只靠聲學模型，長句可能漂移）")
    p.add_argument("--whisper-model", type=str, default="large-v3",
                   help="錨點用的 Whisper 模型，預設 large-v3")
    p.add_argument("--translate-model", type=str, default=None,
                   help="翻譯用的模型，預設 Claude 用 claude-opus-5、OpenAI 用 gpt-4o")
    args = p.parse_args()

    align_opts = {
        'lang': args.lang,
        'separate': not args.no_separate,
        'align_model': args.align_model,
        'anchor': not args.no_anchor,
        'whisper_model': args.whisper_model,
    }

    global TRANSLATE_MODEL
    TRANSLATE_MODEL = args.translate_model

    if args.add:
        if not all([args.title, args.artist, args.video_id, args.lyrics]):
            print("  Required: --title, --artist, --video-id, --lyrics"); sys.exit(1)
        add_song(args.title, args.artist, args.video_id, args.lyrics,
                 args.lyrics_by, args.music_by, args.color, not args.no_translate,
                 args.device, align_opts=align_opts, zh_file=args.zh_file)
    elif args.process_pending:
        process_pending(args.device, not args.no_translate, align_opts=align_opts)
    elif args.import_zh:
        if not args.zh_file: print("  Required: --zh-file"); sys.exit(1)
        import_zh(args.import_zh, args.zh_file)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
