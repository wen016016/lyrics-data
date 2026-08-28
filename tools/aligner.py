# -*- coding: utf-8 -*-
"""
aligner.py - 全域 CTC 強制對齊 (global CTC forced alignment)

治本設計原則
------------
1. 歌詞是「已知的真值」，所以正確做法是 forced alignment，不是「ASR 轉錄 -> 模糊比對歌詞」。
   ASR 在歌聲上錯誤率很高，用它的輸出當定位基準，後面所有階段都只是在補救噪音。

2. 對齊必須是「全曲一次性全域最佳解」。
   把歌曲切塊、逐塊猜視窗往前推進，會累積漂移：一塊錯，後面全錯。
   本模組把「整首歌的所有歌詞」串成一條 token 序列，對「整首歌的所有 frame」跑一次
   CTC Viterbi DP，取全域最佳路徑。沒有 cursor、沒有視窗猜測、沒有塊邊界誤差。

3. 間奏/前奏/尾奏由 CTC blank 自然吸收，不需要事先偵測，也不需要估計唱速。

4. start 與 end 都必須來自音訊 frame，絕對不可以用「字數 / 每秒字數」推算。
   任何依字數合成的時間都會讓字幕在唱到一半時就切掉。

5. 語言無關：文字 -> 對齊 token 的轉換由「該語言的 wav2vec2 vocab」決定，
   不在程式裡硬寫日文假名邏輯。OOV 字元被丟棄時，行號對應表仍然保持正確。

6. 人聲分離 (demucs) 是選用但強烈建議：伴奏與合音會污染 CTC posterior。

實際流水線
----------
    音訊
     -> demucs 抽人聲                    （伴奏/合音不再干擾）
     -> Whisper large-v3 轉錄            （骨架：哪句大概在哪一段）
     -> 全域序列比對 (Needleman-Wunsch)   （歌詞字元 <-> ASR 字元，無門檻、天生單調）
     -> wav2vec2 frame-level emission
     -> 人聲活動遮罩                      （間奏在哪，由 emission 自己說）
     -> 全域 CTC Viterbi + 位置先驗        （精確邊界，且不會漂到間奏）
     -> 能量延伸行尾                      （補回拖長音，字幕不會唱到一半就熄）

為什麼要兩個模型：wav2vec2 CTC 給得出 frame 級邊界，但它是用朗讀語料訓練的，
在歌聲上符號正確率只有三成左右（實測這首歌 30%），純聲學的全域最佳解會讓
證據薄弱的句子把 token 撒到 40 秒的範圍。Whisper 對唱歌辨識好得多，但它的
時間戳是 word/segment 級而且會漂。各取所長：Whisper 定位、wav2vec2 定邊界。
"""

import os
import re
import sys
import unicodedata

import numpy as np

SAMPLE_RATE = 16000
# wav2vec2 conv feature extractor: 每個輸出 frame 對應 320 samples (20ms)
FRAME_STRIDE = 320
FRAME_SEC = FRAME_STRIDE / SAMPLE_RATE
# conv stack 的第一個 frame 覆蓋 400 samples
FRAME_RECEPTIVE = 400

# 每個語言的 CTC 對齊模型。key 是 whisper/ISO 語言碼。
HF_ALIGN_MODELS = {
    'ja': 'jonatasgrosman/wav2vec2-large-xlsr-53-japanese',
    'en': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
    'zh': 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
    'ko': 'kresnik/wav2vec2-large-xlsr-korean',
    'fr': 'jonatasgrosman/wav2vec2-large-xlsr-53-french',
    'de': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
    'es': 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
    'it': 'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
    'pt': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
    'nl': 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch',
    'ru': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    'pl': 'jonatasgrosman/wav2vec2-large-xlsr-53-polish',
    'ar': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
    'fi': 'jonatasgrosman/wav2vec2-large-xlsr-53-finnish',
    'el': 'jonatasgrosman/wav2vec2-large-xlsr-53-greek',
    'hu': 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian',
    'fa': 'jonatasgrosman/wav2vec2-large-xlsr-53-persian',
    'tr': 'mpoyraz/wav2vec2-xls-r-300m-cv7-turkish',
    'vi': 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    'id': 'cahya/wav2vec2-large-xlsr-indonesian',
    'sv': 'KBLab/wav2vec2-large-voxrex-swedish',
}

# 使用空白字元分詞的語言（會餵入 word delimiter token）。
# 日文/中文歌詞裡的空白是排版用的，不是詞界，餵進去會拖累對齊。
SPACE_DELIMITED = {
    'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'ru', 'pl', 'ar', 'fi',
    'el', 'hu', 'fa', 'tr', 'vi', 'id', 'sv', 'uk', 'cs', 'ro', 'ca',
}


# ---------------------------------------------------------------- 語言判斷

def detect_language(plains, override=None):
    """從歌詞本身的文字系統判斷語言。

    歌詞是使用者提供的真值，用它判斷語言比跑一次 Whisper 便宜也可靠得多
    （而且不用為了偵測語言就付一次完整轉錄的成本）。
    """
    if override:
        return override.lower()

    text = ''.join(plains)
    counts = {'kana': 0, 'han': 0, 'hangul': 0, 'latin': 0, 'cyrillic': 0}
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            counts['kana'] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            counts['han'] += 1
        elif 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF:
            counts['hangul'] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts['cyrillic'] += 1
        elif ch.isalpha() and o < 0x0250:
            counts['latin'] += 1

    if counts['kana'] >= 3:
        return 'ja'
    if counts['hangul'] >= 3:
        return 'ko'
    if counts['han'] >= 3:
        return 'zh'
    if counts['cyrillic'] > counts['latin']:
        return 'ru'
    return 'en'


# ---------------------------------------------------------------- 文字正規化

_BRACKETS = re.compile(r'[（）()「」『』【】〔〕\[\]{}《》〈〉]')
_PUNCT = re.compile(r'[、。，．,\.!！?？…‥・:：;；\-–—~〜=+*/\\|"“”\'’`^&%$#@_]')


def _normalize_text(text, lang):
    """去掉不發音的排版符號，並統一全形/半形。"""
    text = unicodedata.normalize('NFKC', text)
    text = _BRACKETS.sub(' ', text)
    text = _PUNCT.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _to_align_script(text, lang, kks=None, vocab=None):
    """把一行歌詞轉成該語言 CTC 模型看得懂的書寫形式。

    ja: 以 vocab 為準 —— 字面在 vocab 裡就用字面（這顆模型是用漢字混假名的
        轉錄訓練的，vocab 裡有 1800 多個漢字，實測用字面的命中率比全轉平假名高），
        只有 vocab 沒收的字才退回讀音。
    其他語言: 原文即可，實際過濾交給 vocab 驅動的逐字比對。
    """
    if lang == 'zh':
        return to_simplified(text)
    if lang != 'ja' or kks is None:
        return text

    out = []
    for item in kks.convert(text):
        orig = item['orig']
        if vocab is None or all(c in vocab or c.isspace() for c in orig):
            out.append(orig)
        else:
            out.append(item['hira'])
    return ''.join(out)


# ---------------------------------------------------- 建立對齊 target 序列

def build_targets(plains, lang, vocab, blank_id, word_delim_id):
    """把每一行歌詞轉成 CTC target token id，並記錄每個 token 屬於哪一行。

    重點：OOV 字元被丟棄時，token -> 行號 的對應表仍然正確。
    舊版程式假設「每個輸入字元剛好產生一個對齊字元」，一旦模型丟掉 OOV
    （日文歌裡的英文單字就會），之後所有行的邊界索引就整體錯開。
    這裡改成明確攜帶行號，結構上不可能再發生那個錯位。
    """
    import pykakasi
    kks = pykakasi.kakasi() if lang == 'ja' else None

    use_delim = lang in SPACE_DELIMITED and word_delim_id is not None

    targets = []       # token id
    token_line = []    # 該 token 屬於哪一行
    per_line_kept = [0] * len(plains)
    dropped = {}       # 被丟棄的字元 -> 次數

    for li, raw in enumerate(plains):
        text = _normalize_text(raw, lang)
        text = _to_align_script(text, lang, kks, vocab)

        prev_was_delim = True
        for ch in text:
            if ch.isspace():
                if use_delim and not prev_was_delim:
                    targets.append(word_delim_id)
                    token_line.append(li)
                    prev_was_delim = True
                continue

            tid = None
            for cand in (ch, ch.upper(), ch.lower()):
                if cand in vocab:
                    tid = vocab[cand]
                    break
            if tid is None or tid == blank_id:
                dropped[ch] = dropped.get(ch, 0) + 1
                continue

            targets.append(tid)
            token_line.append(li)
            per_line_kept[li] += 1
            prev_was_delim = False

        # 行尾不留下懸空的分詞符號
        while targets and use_delim and targets[-1] == word_delim_id:
            targets.pop()
            token_line.pop()

    return targets, token_line, per_line_kept, dropped


# ---------------------------------------------------------------- 人聲分離

def separate_vocals(audio, device='cpu', verbose=True):
    """用 demucs 抽出人聲軌。失敗就回傳原始混音。

    伴奏與合音會直接污染 CTC posterior；在分離後的人聲上對齊，
    間奏段落才會真的接近靜音，讓 blank 乾淨地吃掉它。
    """
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except Exception:
        if verbose:
            print("  [align] demucs 未安裝，直接用混音對齊"
                  "（有合音/厚伴奏的歌建議安裝: pip install demucs）", flush=True)
        return audio, False

    try:
        import torch
        if verbose:
            print("  [align] 人聲分離 (demucs)...", flush=True)
        model = get_model('htdemucs')
        model.to(device).eval()

        # demucs 需要 44.1kHz 立體聲
        import torchaudio
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        wav44 = torchaudio.functional.resample(wav, SAMPLE_RATE, model.samplerate)
        wav44 = wav44.repeat(2, 1)  # mono -> fake stereo

        ref = wav44.mean(0)
        wav44 = (wav44 - ref.mean()) / (ref.std() + 1e-8)

        with torch.inference_mode():
            est = apply_model(model, wav44.unsqueeze(0), device=device,
                              split=True, overlap=0.1, progress=verbose)[0]
        est = est * ref.std() + ref.mean()

        idx = model.sources.index('vocals')
        vocals = est[idx].mean(0, keepdim=True)
        vocals16 = torchaudio.functional.resample(vocals, model.samplerate, SAMPLE_RATE)
        out = vocals16.squeeze(0).numpy().astype(np.float32)

        # 對齊長度，避免 frame 索引位移
        if len(out) < len(audio):
            out = np.pad(out, (0, len(audio) - len(out)))
        out = out[:len(audio)]
        if verbose:
            print("  [align] 人聲分離完成", flush=True)
        return out, True
    except Exception as e:
        if verbose:
            print(f"  [align] 人聲分離失敗 ({e})，改用混音", flush=True)
        return audio, False


# ---------------------------------------------------------------- Emission

def load_ctc_model(name, device='cpu'):
    """載入 CTC 對齊模型的 tokenizer / feature extractor / 模型。

    刻意不用 AutoProcessor：有些模型（例如 xlsr-53-english）在 repo 裡放的是
    Wav2Vec2ProcessorWithLM，AutoProcessor 會去要 pyctcdecode 而直接 ImportError。
    我們只需要 vocab 和特徵正規化設定，不需要它的 beam search 解碼器。
    """
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
    except Exception:
        from transformers import Wav2Vec2CTCTokenizer
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(name)

    feature_extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForCTC.from_pretrained(name).to(device).eval()
    return tokenizer, feature_extractor, model


def compute_emissions(audio, model, feature_extractor, device='cpu',
                      keep_tokens=None, window_sec=20.0, context_sec=2.0,
                      verbose=True):
    """對整首歌算出 frame-level log-probs。

    用重疊視窗推論、只保留每個視窗的內部 frame，所以結果等價於「一次跑完整首歌」，
    但記憶體是有界的。視窗起點對齊到 FRAME_STRIDE 的倍數，
    因此 global_frame = seg_start // 320 + local_frame 是精確對應，不是估計。

    keep_tokens: 只保留這些 vocab 欄位（log_softmax 在完整 vocab 上算完才切），
                 讓記憶體從 T x 2341 降到 T x ~90。
    """
    import torch

    n = len(audio)
    total_frames = max(1, (n - FRAME_RECEPTIVE) // FRAME_STRIDE + 1)

    do_norm = getattr(feature_extractor, 'do_normalize', True)
    a = audio.astype(np.float32)
    if do_norm:
        a = (a - a.mean()) / (a.std() + 1e-7)

    if keep_tokens is None:
        vocab_size = model.config.vocab_size
        col_index = None
        out_dim = vocab_size
    else:
        col_index = torch.tensor(keep_tokens, dtype=torch.long)
        out_dim = len(keep_tokens)

    emission = np.full((total_frames, out_dim), -30.0, dtype=np.float32)

    win = int(window_sec * SAMPLE_RATE)
    ctx = int(context_sec * SAMPLE_RATE)
    win = (win // FRAME_STRIDE) * FRAME_STRIDE
    ctx = (ctx // FRAME_STRIDE) * FRAME_STRIDE

    pos = 0
    n_windows = max(1, (n + win - 1) // win)
    wi = 0
    while pos < n:
        seg_start = max(0, pos - ctx)
        seg_start = (seg_start // FRAME_STRIDE) * FRAME_STRIDE
        seg_end = min(n, pos + win + ctx)
        seg = a[seg_start:seg_end]
        if len(seg) < FRAME_RECEPTIVE:
            break

        with torch.inference_mode():
            iv = torch.from_numpy(seg).unsqueeze(0).to(device)
            logits = model(iv).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            if col_index is not None:
                lp = lp.index_select(1, col_index.to(lp.device))
            lp = lp.cpu().numpy()

        frame_off = seg_start // FRAME_STRIDE
        keep_lo = pos // FRAME_STRIDE                     # 這個視窗負責的第一個 frame
        keep_hi = min(total_frames, (min(n, pos + win)) // FRAME_STRIDE)
        for gf in range(keep_lo, keep_hi):
            lf = gf - frame_off
            if 0 <= lf < lp.shape[0]:
                emission[gf] = lp[lf]

        wi += 1
        if verbose:
            print(f"\r  [align] emission {wi}/{n_windows} windows", end='', flush=True)
        pos += win

    if verbose:
        print(f"\r  [align] emission: {total_frames} frames "
              f"({total_frames * FRAME_SEC:.1f}s)          ", flush=True)
    return emission


# ------------------------------------------------------- CTC forced alignment

def ctc_forced_align(emission, targets, blank_col,
                     expected_frames=None, tol_sec=4.0, penalty_per_sec=1.2):
    """全域 CTC Viterbi 強制對齊。

    emission: (T, C) log-probs，C 已經是壓縮後的欄位
    targets:  (S,) 已經映射到壓縮欄位的 token 索引
    回傳每個 target token 的 (start_frame, end_frame, mean_logprob)

    自己實作而不用 torchaudio.functional.forced_align，因為那支 API 在
    torchaudio 2.9 會被移除，而且自己算才能控制 blank 與位置先驗。

    expected_frames: (S,) 每個 token 的預期 frame（來自 Whisper 錨點）。
      給了就啟用位置先驗：偏離預期位置超過 tol_sec 之後，每秒罰 penalty_per_sec。
      這是「軟」約束，不會讓路徑變成無解。

      為什麼需要它：這顆聲學模型在歌聲上的符號正確率只有三成左右，
      純聲學的全域最佳解會讓證據薄弱的句子把 token 撒到 40 秒的範圍裡。
      位置先驗把 Whisper 的證據帶進來當骨架，聲學模型只負責它擅長的事 ——
      在骨架附近給出 frame 級的精確邊界。
      注意這個先驗來自「音訊證據」，不是「字數推算」。
    """
    T, C = emission.shape
    S = len(targets)
    if S == 0 or T == 0:
        return []

    # 擴展序列: blank, t0, blank, t1, blank, ...
    ext = np.empty(2 * S + 1, dtype=np.int64)
    ext[0::2] = blank_col
    ext[1::2] = np.asarray(targets, dtype=np.int64)
    L = len(ext)

    if T < S:
        raise ValueError(f"音訊 frame 數 ({T}) 少於歌詞 token 數 ({S})，無法對齊")

    # emit[t, s] = 在時間 t 發出 ext[s] 的 log-prob
    emit = emission[:, ext]  # (T, L) float32

    NEG = -1e30
    # 允許 s -> s-2 的跳躍（跳過 blank），條件是 ext[s] 非 blank 且與 ext[s-2] 不同
    can_skip = np.zeros(L, dtype=bool)
    if L > 2:
        idx = np.arange(2, L)
        can_skip[2:] = (ext[2:] != blank_col) & (ext[2:] != ext[:-2])

    # 位置先驗：把每個 token 的預期 frame 擴展到 blank 狀態上
    exp_ext = None
    if expected_frames is not None:
        ef = np.asarray(expected_frames, dtype=np.float32)
        exp_ext = np.empty(L, dtype=np.float32)
        exp_ext[1::2] = ef
        exp_ext[0] = ef[0]
        exp_ext[2::2] = ef  # blank[s] 沿用它前面那個 token 的預期位置
        tol = tol_sec / FRAME_SEC
        lam = penalty_per_sec * FRAME_SEC

    def pos_pen(t):
        if exp_ext is None:
            return 0.0
        return -lam * np.maximum(0.0, np.abs(t - exp_ext) - tol)

    alpha = np.full(L, NEG, dtype=np.float32)
    alpha[0] = emit[0, 0]
    if L > 1:
        alpha[1] = emit[0, 1]
    if exp_ext is not None:
        p0 = pos_pen(0)
        alpha[0] += p0[0]
        if L > 1:
            alpha[1] += p0[1]

    # backtrack: 每個 (t, s) 記錄前一個狀態相對位移 0/1/2
    back = np.zeros((T, L), dtype=np.int8)

    arange_L = np.arange(L)
    for t in range(1, T):
        stay = alpha
        move1 = np.concatenate(([NEG], alpha[:-1]))
        move2 = np.full(L, NEG, dtype=np.float32)
        if L > 2:
            m2 = np.concatenate(([NEG, NEG], alpha[:-2]))
            move2 = np.where(can_skip, m2, NEG).astype(np.float32)

        stacked = np.vstack((stay, move1, move2))
        choice = np.argmax(stacked, axis=0).astype(np.int8)
        best = stacked[choice, arange_L]
        alpha = best + emit[t] + pos_pen(t)
        back[t] = choice

    # 終點只能落在最後一個 token 或其後的 blank
    if alpha[L - 1] >= alpha[L - 2]:
        s = L - 1
    else:
        s = L - 2

    # 回溯出每個 frame 的狀態
    path = np.empty(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s -= int(back[t, s])
            if s < 0:
                s = 0

    # 把狀態序列轉成每個 target token 的 frame 區間
    spans = [None] * S
    for t in range(T):
        s = int(path[t])
        if s % 2 == 0:
            continue  # blank
        ti = s // 2
        if spans[ti] is None:
            spans[ti] = [t, t + 1, float(emit[t, s]), 1]
        else:
            spans[ti][1] = t + 1
            spans[ti][2] += float(emit[t, s])
            spans[ti][3] += 1

    out = []
    for ti in range(S):
        sp = spans[ti]
        if sp is None:
            out.append(None)
        else:
            out.append((sp[0], sp[1], sp[2] / max(1, sp[3])))
    return out


# ------------------------------------------------------------ Whisper 錨點

def _phonetic_key(text, lang, kks=None):
    """把文字轉成用來做序列比對的音韻鍵。

    ja: 統一成平假名（漢字讀音也展開），這樣「漢字寫法」和「假名寫法」可以互相比對。
    zh: 統一成簡體（Whisper 中文預設吐簡體，歌詞常是繁體，不統一就完全比不上）。
    其他語言: 小寫化，只留字母數字。

    注意這裡用 str.isalnum() 而不是 [a-z0-9] 白名單 —— 後者會把漢字、諺文、
    假名全部濾掉，中文與韓文歌的錨點會整段變成空字串，錨點階段直接失效。
    """
    text = _normalize_text(text, lang)
    if lang == 'ja' and kks is not None:
        text = ''.join(item['hira'] for item in kks.convert(text))
        # 片假名一律轉平假名，長音符號保留（它承載時長資訊）
        out = []
        for ch in text:
            o = ord(ch)
            if 0x30A1 <= o <= 0x30F6:
                out.append(chr(o - 0x60))
            else:
                out.append(ch)
        return ''.join(c for c in out if not c.isspace())
    if lang == 'zh':
        text = to_simplified(text)
    return ''.join(c.lower() for c in text if c.isalnum())


_T2S = None


def to_simplified(text):
    """繁體轉簡體。沒安裝 opencc 就原樣回傳（並在第一次呼叫時提示）。

    中文的 CTC 對齊模型（xlsr-53-chinese-zh-cn）和 Whisper 的中文輸出都是簡體，
    繁體歌詞不轉換的話，實測 16 個字裡有 6 個直接不在 vocab 裡。
    """
    global _T2S
    if _T2S is None:
        try:
            import opencc
            _T2S = opencc.OpenCC('t2s')
        except Exception:
            print("  [align] 提示: 未安裝 opencc，繁體中文歌詞的對齊會變差。"
                  "建議 pip install opencc-python-reimplemented", flush=True)
            _T2S = False
    if _T2S is False:
        return text
    try:
        return _T2S.convert(text)
    except Exception:
        return text


def needleman_wunsch(a, b, match=2.0, mismatch=-1.0, gap=-1.0):
    """全域序列比對，回傳 a 的每個位置對到 b 的哪個位置（對不到給 None）。

    取代舊版的「滑動視窗 LCS + beam search + 兩輪 MISS 補救」。
    那套做法有 0.30 / 0.18 / 0.14 等一堆比對門檻，等於幾乎任何位置都能匹配；
    全域最佳比對不需要任何門檻，而且天生保證單調不交錯。
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return [None] * n

    # 只保留分數矩陣的兩列，回溯用 int8 方向矩陣
    score_prev = np.arange(m + 1, dtype=np.float32) * gap
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1=diag 2=up(a gap b) 3=left
    ptr[0, 1:] = 3
    ptr[1:, 0] = 2

    b_arr = np.frombuffer(b.encode('utf-32-le'), dtype=np.uint32)
    for i in range(1, n + 1):
        ai = ord(a[i - 1])
        score_cur = np.empty(m + 1, dtype=np.float32)
        score_cur[0] = i * gap
        eq = np.where(b_arr == ai, match, mismatch).astype(np.float32)
        # 這一列必須逐格算（left 依賴同列前一格）
        prev_left = score_cur[0]
        sp = score_prev
        for j in range(1, m + 1):
            d = sp[j - 1] + eq[j - 1]
            u = sp[j] + gap
            l = prev_left + gap
            if d >= u and d >= l:
                prev_left = d
                ptr[i, j] = 1
            elif u >= l:
                prev_left = u
                ptr[i, j] = 2
            else:
                prev_left = l
                ptr[i, j] = 3
            score_cur[j] = prev_left
        score_prev = score_cur

    mapping = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        d = ptr[i, j]
        if d == 1:
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif d == 2:
            i -= 1
        else:
            j -= 1
    return mapping


def whisper_anchor_times(audio, plains, lang, device='cpu', model_size='large-v3',
                         verbose=True):
    """用 Whisper 轉錄取得「每個歌詞字元的大致時間」。

    Whisper 對唱歌的辨識率遠高於 wav2vec2 CTC 模型（後者是用朗讀語料訓練的，
    在這首歌上符號正確率只有三成），所以它適合當骨架；
    但它的時間戳是 segment/word 級、且會漂，所以不適合直接當字幕邊界。
    兩者分工：Whisper 決定「哪句在哪一段」，wav2vec2 決定「精確到 frame 的邊界」。
    """
    import stable_whisper
    import pykakasi

    kks = pykakasi.kakasi() if lang == 'ja' else None

    if verbose:
        print(f"  [align] Whisper {model_size} 轉錄取錨點 "
              f"(CPU 較慢，可用 --no-anchor 關閉)...", flush=True)
    ct = 'int8' if device == 'cpu' else 'float16'
    model = stable_whisper.load_faster_whisper(model_size, device=device,
                                               compute_type=ct)
    try:
        res = model.transcribe_stable(audio, language=lang, vad=True)
    except Exception as e:
        if verbose:
            print(f"  [align] VAD 失敗 ({e})，改用無 VAD", flush=True)
        res = model.transcribe_stable(audio, language=lang, vad=False)

    words = [{'text': w.word, 'start': w.start, 'end': w.end}
             for seg in res.segments for w in seg.words]
    return anchors_from_words(words, plains, lang, kks, verbose=verbose)


def anchors_from_words(words, plains, lang, kks=None, verbose=True):
    """把 ASR 的詞級時間戳 + 歌詞，變成「每個歌詞字元的大致時間」。

    與轉錄分開，這樣可以拿快取的轉錄結果重跑驗證，不必每次都等 Whisper。
    """
    # ASR 音韻串，並記錄每個字元的時間（在詞內線性插值，
    # 舊版把整個詞的每個字元都標成「詞的開始時間」，時間解析度只有詞級）
    asr_key = []
    asr_time = []
    for w in words:
        k = _phonetic_key(w['text'], lang, kks)
        if not k:
            continue
        dur = max(0.0, w['end'] - w['start'])
        for ci, ch in enumerate(k):
            asr_key.append(ch)
            asr_time.append(w['start'] + dur * (ci + 0.5) / len(k))
    if not asr_key:
        raise RuntimeError("Whisper 沒有產生可用的錨點")

    # 歌詞音韻串 + 每個字元屬於哪一行
    lyr_key = []
    lyr_line = []
    for li, raw in enumerate(plains):
        for ch in _phonetic_key(raw, lang, kks):
            lyr_key.append(ch)
            lyr_line.append(li)

    if verbose:
        print(f"  [align] 全域序列比對 ({len(lyr_key)} 歌詞字元 vs "
              f"{len(asr_key)} ASR 字元)...", flush=True)
    mapping = needleman_wunsch(''.join(lyr_key), ''.join(asr_key))

    matched = sum(1 for m in mapping if m is not None)
    char_time = [None] * len(lyr_key)
    for i, m in enumerate(mapping):
        if m is not None:
            char_time[i] = asr_time[m]

    # 讓時間單調（比對本身單調，但 ASR 時間戳偶爾會逆行）
    last = -1e9
    for i in range(len(char_time)):
        if char_time[i] is None:
            continue
        if char_time[i] < last:
            char_time[i] = last
        last = char_time[i]

    if verbose:
        print(f"  [align] 錨點命中 {matched}/{len(lyr_key)} 歌詞字元 "
              f"({100.0 * matched / max(1, len(lyr_key)):.0f}%)", flush=True)
    return char_time, lyr_line


def expected_frames_from_anchors(anchor_bounds, token_line, n_lines,
                                 total_frames):
    """把「每行的錨點起訖」換成「對齊 token -> 預期 frame」。

    直接吃 anchor_line_bounds() 的輸出（已做離群剔除），
    所以位置先驗和最終邊界用的是同一份錨點，不會兩邊不一致。
    沒有錨點的行由前後行插值。
    """
    line_lo = [b[0] if b else None for b in anchor_bounds]
    line_hi = [b[1] if b else None for b in anchor_bounds]

    known = [i for i in range(n_lines) if line_lo[i] is not None]
    if not known:
        return None

    span = total_frames * FRAME_SEC
    for li in range(n_lines):
        if line_lo[li] is not None:
            continue
        prev = max([k for k in known if k < li], default=None)
        nxt = min([k for k in known if k > li], default=None)
        if prev is not None and nxt is not None:
            a, b = line_hi[prev], line_lo[nxt]
            frac = (li - prev) / (nxt - prev)
            line_lo[li] = a + frac * max(0.0, b - a)
            line_hi[li] = a + (frac + 1.0 / (nxt - prev)) * max(0.0, b - a)
        elif prev is not None:
            line_lo[li] = min(span, line_hi[prev] + 0.3)
            line_hi[li] = min(span, line_lo[li] + 2.0)
        else:
            line_lo[li] = max(0.0, line_lo[nxt] - 2.5)
            line_hi[li] = max(0.1, line_lo[nxt] - 0.3)

    # 行內展開到每個 token
    counts = {}
    for li in token_line:
        counts[li] = counts.get(li, 0) + 1
    seen = {}
    out = []
    for li in token_line:
        k = seen.get(li, 0)
        seen[li] = k + 1
        n = counts[li]
        lo, hi = line_lo[li], max(line_hi[li], line_lo[li] + 0.2)
        t = lo + (hi - lo) * (k + 0.5) / n
        out.append(t / FRAME_SEC)
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------- 能量輔助

def frame_rms(audio, total_frames):
    """每個 CTC frame 的 RMS，用來處理拖長音與句首氣音。"""
    n = total_frames * FRAME_STRIDE
    a = audio[:n]
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    a = a.reshape(total_frames, FRAME_STRIDE)
    return np.sqrt((a.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)


def vocal_activity_mask(emission, blank_col, rms=None, win_sec=1.5,
                        min_symbols=2, energy_ratio=0.15):
    """判斷哪些 frame 真的有人聲。

    為什麼需要這個：整首歌的「非 blank frame 數」和「歌詞 token 數」幾乎一樣多
    （這首歌 549 vs 547），所以 DP 幾乎沒有餘裕 —— 它被迫用掉每一個非 blank frame，
    包含間奏裡零星的假訊號。加上 Whisper 偶爾會在間奏裡吐出一兩個字，
    一行的範圍就會從 5 秒被撐成 27 秒橫跨整段間奏。

    判準一（音素密度）：以某 frame 為中心的 ±win_sec 窗內，模型吐出的符號數
    <= min_symbols。真正在唱的段落每 3 秒有 6~12 個符號，間奏是 0~2 個。

    判準二（能量）：人聲軌的 RMS 低於「有聲部分中位數的 energy_ratio」。

    兩個判準必須同時成立才算非人聲。只用密度會誤判拖長音 ——
    CTC 對一個拖 5 秒的長音只吐一個符號，密度跟間奏一樣低，
    但那裡明明有人在唱。實測這首歌就有兩處（約 17-30s 與 200-210s）
    是「音量跟副歌一樣大、但幾乎沒有音素」的長音段落。

    rms=None 時退回只用密度判準（沒做人聲分離時，混音的能量分不出間奏與人聲，
    能量判準會讓遮罩失效，所以只能靠密度，但拖長音就有被誤剪的風險）。
    """
    nonblank = (emission.argmax(axis=1) != blank_col).astype(np.float32)
    k = int(win_sec / FRAME_SEC)
    dens = np.convolve(nonblank, np.ones(2 * k + 1, dtype=np.float32), mode='same')
    low_density = dens <= min_symbols

    if rms is None:
        return ~low_density

    dense_rms = rms[~low_density]
    if len(dense_rms) < 10:
        return ~low_density
    thr = float(np.median(dense_rms)) * energy_ratio
    low_energy = rms < thr
    return ~(low_density & low_energy)


def apply_nonvocal_prior(emission, vocal_mask, blank_col, boost=6.0, verbose=True):
    """在判定沒有唱字的 frame 上加強 blank，把 token 擠出間奏。

    用「加權」而不是「硬禁止」：長拖音那種「音量大但沒有音素」的段落也會被
    這個 mask 標成非人聲，留一條逃生路徑比較安全（拖音本身的長度由
    extend_end_by_energy 負責，不需要靠 token 佔位）。
    """
    off = ~vocal_mask
    if off.sum() == 0:
        return emission
    emission[off, blank_col] += boost
    if verbose:
        runs = _mask_runs(off, min_sec=2.0)
        desc = ', '.join(f"{a:.0f}-{b:.0f}s" for a, b in runs[:8])
        print(f"  [align] 非人聲段落 {off.sum() * FRAME_SEC:.0f}s"
              f"{(' (' + desc + ')') if desc else ''}", flush=True)
    return emission


def _mask_runs(mask, min_sec=2.0):
    """把 boolean mask 轉成連續區間 (秒)，只回報長度超過 min_sec 的。"""
    runs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - start) * FRAME_SEC >= min_sec:
                runs.append((start * FRAME_SEC, i * FRAME_SEC))
            start = None
    if start is not None and (len(mask) - start) * FRAME_SEC >= min_sec:
        runs.append((start * FRAME_SEC, len(mask) * FRAME_SEC))
    return runs


# ---------------------------------------------------------------- 邊界後處理

def clampable_mask(vocal_mask, min_gap_sec=1.5):
    """把「短暫的非人聲空隙」填回人聲，只留下夠長的間奏當裁切邊界。

    句與句之間本來就有換氣與短停頓，那些不該被當成間奏來裁句子。
    只有持續超過 min_gap_sec 的段落才算真正的間奏。
    """
    m = vocal_mask.copy()
    k = int(min_gap_sec / FRAME_SEC)
    start = None
    for i, val in enumerate(m):
        if not val and start is None:
            start = i
        elif val and start is not None:
            if i - start < k:
                m[start:i] = True
            start = None
    if start is not None and len(m) - start < k:
        m[start:] = True
    return m


def clamp_to_vocal(start_f, end_f, cmask, nxt_vocal, prev_vocal,
                   min_keep_frames=20):
    """把一行的邊界收進真正有人聲的區間。

    這是最後一個大誤差來源：Whisper 偶爾會在間奏裡吐出一兩個字，
    被錨點吃進去之後，那一行的範圍就從 5 秒被撐成 27 秒，橫跨整段間奏。
    裁切的依據是聲學模型自己判定的人聲活動，不是猜的。

    裁切後長度不足 min_keep_frames 時回傳 None，表示「這行的錨點整段都不在人聲上」
    （Whisper 在間奏裡幻聽的典型結果），交給 relocate_into_vocal() 重新安置。
    """
    s, e = start_f, end_f
    if e <= s:
        return s, e

    # 整段都沒有人聲：兩端各自往內收也救不回來（收出來的區間會互相跨過），
    # 必須明確回報無效，交給重新安置處理。
    if not cmask[s:e].any():
        return None

    ns, ne = s, e
    if not cmask[ns]:
        g = nxt_vocal[ns]
        if g >= 0 and g < ne:
            ns = g
    if not cmask[ne - 1]:
        g = prev_vocal[ne - 1]
        if g >= 0 and g >= ns:
            ne = g + 1

    if ne - ns < min_keep_frames:
        return None
    return ns, max(ne, ns + 1)


def relocate_into_vocal(lo, hi, cmask):
    """在 [lo, hi) 之間找最長的連續人聲區間。

    給「錨點整段落在間奏裡」的行用：Whisper 在間奏幻聽時，那一行的錨點會
    整段無效，但前後兩行之間通常剛好還剩一段沒被指派的人聲 —— 那才是這行的位置。
    比直接在前後行之間做線性插值可靠得多，因為這是音訊上真的有人在唱的區間。
    """
    if hi <= lo:
        return None
    best = None
    cur = None
    for f in range(lo, hi):
        if cmask[f]:
            if cur is None:
                cur = f
        elif cur is not None:
            if best is None or (f - cur) > (best[1] - best[0]):
                best = (cur, f)
            cur = None
    if cur is not None and (best is None or (hi - cur) > (best[1] - best[0])):
        best = (cur, hi)
    return best


def _vocal_lookups(cmask):
    """預先算好「往後第一個人聲 frame」與「往前第一個人聲 frame」。"""
    n = len(cmask)
    nxt = np.full(n, -1, dtype=np.int32)
    prv = np.full(n, -1, dtype=np.int32)
    last = -1
    for i in range(n):
        if cmask[i]:
            last = i
        prv[i] = last
    nextv = -1
    for i in range(n - 1, -1, -1):
        if cmask[i]:
            nextv = i
        nxt[i] = nextv
    return nxt, prv


def extend_end(end_frame, next_start_frame, rms, vocal_mask, line_rms,
               max_extend_frames, floor):
    """把行尾往後延伸到人聲真的停下來為止。

    wav2vec2 的 CTC posterior 是 peaky 的：最後一個 token 的 frame 落在
    「聽得出那個音」的位置，但唱歌常常把尾音拖 1~2 秒。直接用 CTC 的 end，
    字幕就會在人還在唱的時候熄掉 —— 那正是「前一句還沒結束就被切到下一句」
    的觀感來源。

    停止條件同時看兩件事：音量掉下去，或離開有唱字的區域。
    """
    thr = max(floor, line_rms * 0.30)
    limit = min(next_start_frame, end_frame + max_extend_frames, len(rms))
    f = end_frame
    quiet = 0
    while f < limit:
        if rms[f] >= thr and vocal_mask[f]:
            quiet = 0
        else:
            quiet += 1
            if quiet >= 8:  # 連續 160ms 才認定真的結束
                break
        f += 1
    return max(end_frame, f - quiet)


def extend_start(start_frame, prev_end_frame, rms, vocal_mask, line_rms,
                 max_extend_frames, floor):
    """把行首往前推到能量開始上升的位置（補回被 CTC 吃掉的起音）。"""
    thr = max(floor, line_rms * 0.35)
    limit = max(prev_end_frame, start_frame - max_extend_frames, 1)
    f = start_frame
    while f > limit and rms[f - 1] >= thr and vocal_mask[f - 1]:
        f -= 1
    return f


def _vocal_runs_in(lo, hi, cmask):
    """回傳 [lo, hi) 之間所有連續的人聲區段 [(s, e), ...]。"""
    runs = []
    cur = None
    for f in range(max(0, lo), min(len(cmask), hi)):
        if cmask[f]:
            if cur is None:
                cur = f
        elif cur is not None:
            runs.append((cur, f))
            cur = None
    if cur is not None:
        runs.append((cur, min(len(cmask), hi)))
    return runs


# 判定「這行的區間短到不合理」的字每秒門檻。
# 注意用途：只用來『偵測異常』與『決定要不要重新分配』，
# 絕對不用來『產生』時間 —— 重新分配後的邊界仍然落在真實的人聲 frame 上。
# 這跟舊版「end = start + 字數/速率」是不同性質的東西。
CRAMPED_RATE = 5.0      # 超過這個字每秒，視為候選（可能被擠壓）
MIN_FREE_SEC = 2.0      # 空隙裡至少要有這麼多「人聲」才算可疑


def redistribute_unclaimed_vocal(line_start_f, line_end_f, source, char_counts,
                                 cmask, total_frames, verbose=True):
    """把「沒有任何一行認領、但確實有人在唱」的區段還給被擠壓的句子。

    為什麼需要：Whisper 的轉錄不是完全可重現的，偶爾會把某幾句的錨點整體
    往後拖，結果是「前面留下一段沒人認領的人聲」加上「後面幾句被擠成 0.7 秒」。
    實測 Trinity Field 就出現過 L14 唱完後有 6 秒人聲沒人認領，
    而 L15/L16 被壓在後面 3 秒內（15.3 字/秒，不可能唱得完）。

    判準完全來自音訊：空隙裡有沒有人聲（遮罩說的）、以及那幾句是不是被壓縮。
    重新分配時，先按「人聲區段的長度」把句子分配到區段，再在區段內按字數比例鋪開，
    所以句子不會跨過間奏。
    """
    n = len(line_start_f)
    min_free = int(MIN_FREE_SEC / FRAME_SEC)
    moved = []

    i = 0
    while i < n - 1:
        if line_start_f[i] is None or line_end_f[i] is None:
            i += 1
            continue

        # 找下一個有落點的行
        nxt = None
        for j in range(i + 1, n):
            if line_start_f[j] is not None:
                nxt = j
                break
        if nxt is None:
            break

        gap_lo, gap_hi = line_end_f[i], line_start_f[nxt]
        free = sum(e - s for s, e in _vocal_runs_in(gap_lo, gap_hi, cmask))
        if free < min_free:
            i = nxt
            continue

        # 空隙後面連續被擠壓的句子，就是候選要往前搬的那一批
        run_end = nxt - 1
        for j in range(nxt, n):
            if line_start_f[j] is None or line_end_f[j] is None:
                break
            dur = max(1, line_end_f[j] - line_start_f[j]) * FRAME_SEC
            if char_counts[j] / dur <= CRAMPED_RATE:
                break
            run_end = j
        if run_end < nxt:
            i = nxt
            continue

        # 可用範圍：空隙起點 ~ 這批句子後面第一個沒被擠壓的句子起點
        after = None
        for j in range(run_end + 1, n):
            if line_start_f[j] is not None:
                after = line_start_f[j]
                break
        if after is None:
            after = total_frames

        runs = _vocal_runs_in(gap_lo, after, cmask)
        avail = sum(e - s for s, e in runs)
        current = sum(max(1, line_end_f[k] - line_start_f[k])
                      for k in range(nxt, run_end + 1))
        # 只有在「明顯多出空間」時才動，避免為了幾百毫秒把好的結果攪亂
        if not runs or avail < current * 1.3:
            i = nxt
            continue

        idxs = list(range(nxt, run_end + 1))
        if _lay_out_on_runs(idxs, runs, char_counts, line_start_f, line_end_f):
            for k in idxs:
                source[k] = 'moved'
            moved.extend(idxs)

        i = run_end + 1

    if verbose and moved:
        print(f"  [align] 有未認領的人聲、且後面句子被擠壓，已重新分配的行: "
              f"{[k + 1 for k in moved]}", flush=True)
    return moved


def _lay_out_on_runs(idxs, runs, char_counts, line_start_f, line_end_f):
    """把 idxs 這幾行鋪到 runs 這些人聲區段上。

    先按區段長度的比例決定每個區段接幾行（所以句子不會跨越間奏），
    再在區段內按字數比例切開。
    """
    total_len = sum(e - s for s, e in runs)
    total_chars = sum(max(1, char_counts[k]) for k in idxs)
    if total_len <= 0 or total_chars <= 0:
        return False

    # 分配行到區段：依區段長度換算出「這個區段容納得下多少字」
    groups = []
    pos = 0
    for ri, (s, e) in enumerate(runs):
        if pos >= len(idxs):
            break
        remaining_runs = len(runs) - ri - 1
        cap = (e - s) / total_len * total_chars
        take = []
        acc = 0
        while pos < len(idxs):
            c = max(1, char_counts[idxs[pos]])
            # 至少收一行；收滿容量就換下一個區段，但要留夠行數給後面的區段。
            # 這裡用 >= 而不是 >：剩 1 行、剩 1 個區段時要把那行讓給下一個區段，
            # 否則後面的人聲區段會沒人認領 —— 那正是這個函式要修的毛病。
            if take and acc + c / 2 > cap and (len(idxs) - pos) >= remaining_runs:
                break
            take.append(idxs[pos])
            acc += c
            pos += 1
        if take:
            groups.append(((s, e), take))
    # 還有沒放進去的行，全部塞到最後一個區段
    if pos < len(idxs):
        if not groups:
            return False
        groups[-1][1].extend(idxs[pos:])

    for (s, e), take in groups:
        w = [max(1, char_counts[k]) for k in take]
        tw = sum(w)
        span = e - s
        cum = 0
        for k, wk in zip(take, w):
            share = max(1, int(round(span * wk / tw)))
            a = s + cum
            b = min(e, a + share)
            if b <= a:
                b = min(e, a + 1)
            line_start_f[k] = a
            line_end_f[k] = b
            cum += share
            if s + cum >= e:
                cum = max(0, e - s - 1)
    return True


# ---------------------------------------------------------------- 主流程

def align_lines(audio, plains, align_texts=None, join_prev=None, lang=None,
                device='cpu', separate=True, verbose=True, model_name=None,
                anchor=True, whisper_model='large-v3'):
    """對齊歌詞到音訊，回傳 [{'time','end','score','source'}, ...]

    plains:      顯示用歌詞（只用於報表）
    align_texts: 對齊用歌詞。特殊念法（365 唱成 everyday、日文歌裡的英文段落）
                 在這裡已經被換成實際讀音，所以聲學模型與 Whisper 錨點
                 比對的是「真正唱出來的音」，而不是字面。None 時等同 plains。
    """
    n_lines = len(plains)
    align_texts = list(align_texts) if align_texts else list(plains)
    if len(align_texts) != n_lines:
        raise ValueError("align_texts 行數與歌詞行數不一致")
    audio_len = len(audio) / SAMPLE_RATE
    lang = detect_language(plains, lang)

    name = model_name or HF_ALIGN_MODELS.get(lang)
    if not name:
        raise RuntimeError(f"語言 '{lang}' 沒有對應的對齊模型，"
                           f"請用 --lang 指定支援的語言或 --align-model 指定模型")
    if verbose:
        print(f"  [align] 語言: {lang} / 對齊模型: {name}", flush=True)

    # --- 人聲分離（對合音與厚伴奏的歌最關鍵的一步）---
    vocals, separated = (separate_vocals(audio, device=device, verbose=verbose)
                         if separate else (audio, False))

    # --- 載入 CTC 模型 ---
    tokenizer, feature_extractor, model = load_ctc_model(name, device=device)

    vocab = tokenizer.get_vocab()
    blank_id = tokenizer.pad_token_id
    if blank_id is None:
        blank_id = vocab.get('<pad>', 0)
    delim = getattr(tokenizer, 'word_delimiter_token', '|')
    word_delim_id = vocab.get(delim)

    # --- 歌詞 -> target token（行號對應表結構上保證正確）---
    targets_full, token_line, per_line_kept, dropped = build_targets(
        align_texts, lang, vocab, blank_id, word_delim_id)

    if not targets_full:
        raise RuntimeError("歌詞轉不出任何對齊 token（語言判斷或歌詞內容有問題）")

    empty_lines = [i for i, c in enumerate(per_line_kept) if c == 0]
    if verbose:
        print(f"  [align] 對齊 token: {len(targets_full)} 個 / {n_lines} 行", flush=True)
        if dropped:
            top = sorted(dropped.items(), key=lambda x: -x[1])[:8]
            total = sum(dropped.values())
            print(f"  [align] 模型 vocab 不含而略過 {total} 個字元: "
                  f"{' '.join(repr(c) + 'x' + str(k) for c, k in top)}", flush=True)
        if empty_lines:
            print(f"  [align] 警告: 第 {[i + 1 for i in empty_lines]} 行沒有可對齊字元，"
                  f"將由鄰近行插值", flush=True)

    # --- emission（只保留用得到的 vocab 欄位）---
    used = sorted(set(targets_full) | {blank_id})
    col_of = {tid: i for i, tid in enumerate(used)}
    blank_col = col_of[blank_id]

    emission = compute_emissions(vocals, model, feature_extractor, device=device,
                                 keep_tokens=used, verbose=verbose)
    total_frames = emission.shape[0]

    # --- Whisper 錨點（骨架）---
    expected = None
    anchor_bounds = None
    if anchor:
        try:
            char_time, lyr_line = whisper_anchor_times(
                vocals, align_texts, lang, device=device,
                model_size=whisper_model, verbose=verbose)
            anchor_bounds = anchor_line_bounds(char_time, lyr_line, n_lines,
                                               verbose=verbose)
            expected = expected_frames_from_anchors(
                anchor_bounds, token_line, n_lines, total_frames)
        except Exception as e:
            if verbose:
                print(f"  [align] 錨點階段失敗 ({e})，改用純聲學全域對齊", flush=True)

    tgt_cols = [col_of[t] for t in targets_full]
    char_counts = [len(re.sub(r'\s+', '', p)) for p in plains]
    results = finalize(emission, vocals, tgt_cols, token_line, n_lines,
                       blank_col, audio_len, anchor_bounds=anchor_bounds,
                       expected_frames=expected, separated=separated,
                       char_counts=char_counts, join_prev=join_prev,
                       verbose=verbose)
    return results, lang


def anchor_line_bounds(char_time, lyr_line, n_lines, verbose=True):
    """錨點 -> 每行的 [start, end] 秒。沒命中的行給 None，交給後面插值。

    一行的字元在時間上必然是連續的，所以先做離群剔除：
    Whisper 偶爾會在間奏裡吐出一兩個字，序列比對就把它接到某一行的頭或尾，
    那一行的範圍會從 5 秒被撐成 27 秒（實測 L04 就有一個字落在 22 秒之外）。
    以該行錨點的中位數為中心、容許量隨字數成長，把離群的字元丟掉。
    """
    per_line = [[] for _ in range(n_lines)]
    for i, t in enumerate(char_time):
        if t is not None:
            per_line[lyr_line[i]].append(t)

    bounds = []
    dropped = 0
    for li in range(n_lines):
        ts = per_line[li]
        if not ts:
            bounds.append(None)
            continue
        med = float(np.median(ts))
        # 容許量：一行唱多久本來就跟字數有關，但這裡只用來抓「差一個數量級」的
        # 離群值，不是用來決定時長，所以放得很寬。
        allow = max(4.0, 3.0 + 0.7 * len(ts))
        keep = [t for t in ts if abs(t - med) <= allow]
        if not keep:
            keep = ts
        dropped += len(ts) - len(keep)
        bounds.append((min(keep), max(keep)))

    if verbose and dropped:
        print(f"  [align] 錨點離群剔除: {dropped} 個字元", flush=True)
    return bounds


CTC_TRUST_GLOBAL = 0.45   # 整首歌的 CTC 命中率中位數要多高才信任它的邊界
CTC_TRUST_LINE = 0.40     # 單行的門檻


def finalize(emission, vocals, tgt_cols, token_line, n_lines, blank_col,
             audio_len, anchor_bounds=None, expected_frames=None,
             separated=True, char_counts=None, join_prev=None, verbose=True):
    """emission (+ 錨點) -> 每行的 start/end。與模型推論分離，方便單獨重跑驗證。

    證據分級（實測出來的，不是猜的）：
      Whisper 轉錄 vs 歌詞的符號重疊率  85%（序列比對命中 96% 字元）
      wav2vec2 CTC 轉錄 vs 歌詞的重疊率  30%
    所以邊界以 Whisper 錨點為主，CTC 只有在它「這首歌真的聽懂了」的時候才接手。
    在日文歌上通常是錨點勝出；在拉丁語系歌曲上 wav2vec2 模型強得多，
    這個判斷是量出來的，不是寫死語言規則。
    """
    total_frames = emission.shape[0]
    rms = frame_rms(vocals, total_frames)

    vmask = vocal_activity_mask(emission, blank_col,
                                rms=rms if separated else None)
    cmask = clampable_mask(vmask)
    nxt_vocal, prev_vocal = _vocal_lookups(cmask)
    emission = apply_nonvocal_prior(emission, vmask, blank_col, verbose=verbose)

    # --- 全域 forced alignment：一次算完整首歌 ---
    if verbose:
        kind = "有錨點" if expected_frames is not None else "純聲學"
        print(f"  [align] 全域 CTC 對齊 ({kind}, "
              f"{total_frames} frames x {len(tgt_cols)} tokens)...", flush=True)
    spans = ctc_forced_align(emission, tgt_cols, blank_col,
                             expected_frames=expected_frames)

    # --- token span -> 行邊界 ---
    ctc_start = [None] * n_lines
    ctc_end = [None] * n_lines
    line_scores = [[] for _ in range(n_lines)]
    for ti, sp in enumerate(spans):
        if sp is None:
            continue
        li = token_line[ti]
        s, e, sc = sp
        if ctc_start[li] is None or s < ctc_start[li]:
            ctc_start[li] = s
        if ctc_end[li] is None or e > ctc_end[li]:
            ctc_end[li] = e
        line_scores[li].append(sc)

    hits = [_line_score(line_scores[li]) for li in range(n_lines)]
    hit_vals = sorted(h for h in hits if h is not None)
    global_hit = hit_vals[len(hit_vals) // 2] if hit_vals else 0.0
    trust_ctc = (anchor_bounds is None) or (global_hit >= CTC_TRUST_GLOBAL)
    if verbose:
        print(f"  [align] CTC 命中率中位數 {global_hit:.2f} -> "
              f"邊界主要採用 {'CTC 聲學對齊' if trust_ctc else 'Whisper 錨點'}",
              flush=True)

    # --- 逐行選用最可信的邊界來源 ---
    line_start_f = [None] * n_lines
    line_end_f = [None] * n_lines
    source = [None] * n_lines
    for li in range(n_lines):
        use_ctc = (ctc_start[li] is not None and
                   (trust_ctc or (hits[li] or 0.0) >= CTC_TRUST_LINE))
        if use_ctc:
            line_start_f[li] = ctc_start[li]
            line_end_f[li] = ctc_end[li]
            source[li] = 'ctc'
        elif anchor_bounds is not None and anchor_bounds[li] is not None:
            lo, hi = anchor_bounds[li]
            line_start_f[li] = max(0, int(round(lo / FRAME_SEC)))
            line_end_f[li] = min(total_frames, max(line_start_f[li] + 1,
                                                   int(round(hi / FRAME_SEC))))
            source[li] = 'anchor'

    voiced = rms[rms > 0]
    floor = float(np.percentile(voiced, 20)) if len(voiced) > 10 else 0.0
    max_ext = int(2.0 / FRAME_SEC)
    max_pre = int(0.30 / FRAME_SEC)

    # --- 收進人聲區間，再用能量微調（處理拖長音 / 起音）---
    relocated = []
    for li in range(n_lines):
        if line_start_f[li] is None:
            continue

        nxt_start = total_frames
        for j in range(li + 1, n_lines):
            if line_start_f[j] is not None:
                nxt_start = line_start_f[j]
                break
        prev_end = 0
        for j in range(li - 1, -1, -1):
            if line_end_f[j] is not None:
                prev_end = line_end_f[j]
                break

        clamped = clamp_to_vocal(line_start_f[li], line_end_f[li],
                                 cmask, nxt_vocal, prev_vocal)
        if clamped is None:
            # 錨點整段落在間奏 -> 改放到前後行之間真的有人聲的那一段
            slot = relocate_into_vocal(prev_end, max(prev_end + 1, nxt_start), cmask)
            if slot is None:
                s, e = line_start_f[li], line_end_f[li]
            else:
                s, e = slot
                source[li] = 'moved'
                relocated.append(li)
        else:
            s, e = clamped

        seg = rms[s:max(s + 1, e)]
        line_rms = float(seg.mean()) if len(seg) else 0.0

        line_end_f[li] = extend_end(e, max(s + 1, nxt_start), rms, vmask,
                                    line_rms, max_ext, floor)
        line_start_f[li] = extend_start(s, prev_end, rms, vmask,
                                        line_rms, max_pre, floor)

    if verbose and relocated:
        print(f"  [align] 錨點落在間奏、已重新安置的行: "
              f"{[i + 1 for i in relocated]}", flush=True)

    # 把沒人認領的人聲還給被擠壓的句子（Whisper 錨點整體後移時的修正）
    if char_counts is not None:
        redistribute_unclaimed_vocal(line_start_f, line_end_f, source,
                                     char_counts, cmask, total_frames,
                                     verbose=verbose)

    # --- 組出結果 ---
    results = []
    for li in range(n_lines):
        if line_start_f[li] is None:
            results.append({'time': None, 'end': None, 'score': None,
                            'source': 'interp'})
        else:
            results.append({
                'time': round(line_start_f[li] * FRAME_SEC, 2),
                'end': round(line_end_f[li] * FRAME_SEC, 2),
                'score': hits[li],
                'source': source[li],
            })

    _interpolate_missing(results, audio_len)
    _sanitize(results, audio_len)
    # 合唱：標記過的行與上一句合併成同一個時間範圍，讓播放器同時亮。
    # 刻意放在 _sanitize 之後 —— 前面所有階段都維持「不重疊、單調」的假設，
    # 重疊只在這裡、只對你明確標過的行產生。
    _merge_join_groups(results, join_prev, verbose=verbose)
    return results


def _merge_join_groups(results, join_prev, verbose=True):
    """把「同時唱」的連續行合併成同一個 [time, end] 範圍。

    合併而不是各自給一半：兩個聲部同時唱時，ASR 看到的是交錯的單一串流，
    對齊只能把兩句的 token 前後排開 —— 各自的一半都是錯的，但聯集是對的。
    所以取聯集，讓那段時間內兩句都亮。
    """
    if not join_prev or not any(join_prev):
        return
    n = len(results)
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and j + 1 < len(join_prev) and join_prev[j + 1]:
            j += 1
        if j > i:
            groups.append((i, j))
        i = j + 1

    for a, b in groups:
        lo = min(results[k]['time'] for k in range(a, b + 1))
        hi = max(results[k]['end'] for k in range(a, b + 1))
        for k in range(a, b + 1):
            results[k]['time'] = round(lo, 2)
            results[k]['end'] = round(hi, 2)
    if verbose and groups:
        desc = ', '.join(f"L{a + 1}-L{b + 1}" for a, b in groups)
        print(f"  [align] 合唱合併: {desc}（各組內同時亮）", flush=True)


def _line_score(token_logprobs):
    """這一行有多少比例的 token 真的落在自己的聲學證據上。

    刻意不用 exp(mean(logprob))：CTC 在歌聲上一定有一部分 token 對不上
    （模型漏字、把漢字讀音聽成別的音），那些 token 的 logprob 是 -20 以下，
    平均值會被它們單方面吃掉，讓每一行看起來都像壞掉，失去診斷價值。
    用「命中率」才分得出「這行大致對上」和「這行整段錯位」。
    """
    if not token_logprobs:
        return None
    hit = sum(1 for lp in token_logprobs if lp > -2.3)  # prob > 0.10
    return round(hit / len(token_logprobs), 3)


def _interpolate_missing(results, audio_len):
    """只有「完全沒有可對齊字元」的行才會走到這裡（例如整行是模型 vocab 外的外文）。"""
    n = len(results)
    known = [i for i, r in enumerate(results) if r['time'] is not None]
    if not known:
        for i, r in enumerate(results):
            r['time'] = round(i * audio_len / n, 2)
            r['end'] = round((i + 1) * audio_len / n, 2)
        return

    for i, r in enumerate(results):
        if r['time'] is not None:
            continue
        prev = max([k for k in known if k < i], default=None)
        nxt = min([k for k in known if k > i], default=None)
        if prev is not None and nxt is not None:
            a = results[prev]['end']
            b = results[nxt]['time']
            frac = (i - prev) / (nxt - prev)
            span = max(0.0, b - a)
            r['time'] = round(a + frac * span, 2)
            r['end'] = round(a + (frac + 1.0 / (nxt - prev)) * span, 2)
        elif prev is not None:
            a = results[prev]['end']
            r['time'] = round(min(audio_len, a + 0.2), 2)
            r['end'] = round(min(audio_len, a + 1.5), 2)
        else:
            b = results[nxt]['time']
            r['time'] = round(max(0.0, b - 1.7), 2)
            r['end'] = round(max(0.05, b - 0.2), 2)


def _sanitize(results, audio_len):
    """最小必要的清理。

    刻意「不做」的事：不依字數合成時長、不因為前一行的推算長度去推遲下一行的開始。
    那種做法會把時間軸變成字數節拍器，也是原本字幕早切的直接原因。
    這裡只保證：在音訊範圍內、start < end、相鄰行不重疊（重疊時裁前一行的尾，
    不動下一行由音訊決定的 start）。
    """
    n = len(results)
    for r in results:
        r['time'] = max(0.0, min(audio_len, r['time']))
        r['end'] = max(0.0, min(audio_len, r['end']))

    for i in range(n):
        if results[i]['end'] < results[i]['time'] + 0.12:
            results[i]['end'] = min(audio_len, results[i]['time'] + 0.12)

    for i in range(1, n):
        if results[i]['time'] < results[i - 1]['time']:
            results[i]['time'] = results[i - 1]['time']
        if results[i - 1]['end'] > results[i]['time'] - 0.02:
            trimmed = results[i]['time'] - 0.02
            results[i - 1]['end'] = max(results[i - 1]['time'] + 0.12, trimmed)

    for r in results:
        r['time'] = round(r['time'], 2)
        r['end'] = round(r['end'], 2)


def report(results, plains, verbose=True):
    """列出每行的時間、時長、語速與信賴度，方便肉眼抓出可疑的行。"""
    if not verbose:
        return
    print("  " + "-" * 78, flush=True)
    print(f"  {'行':>3} {'start':>7} {'end':>7} {'dur':>6} {'字/秒':>6} "
          f"{'來源':>6} {'gap':>6}", flush=True)
    prev_end = None
    low = []
    for i, r in enumerate(results):
        dur = r['end'] - r['time']
        nchars = len(re.sub(r'\s+', '', plains[i]))
        rate = nchars / dur if dur > 0.05 else 0.0
        gap = (r['time'] - prev_end) if prev_end is not None else 0.0

        # 只標記「時間軸本身可疑」的行。
        # CTC 命中率低不代表這行有問題 —— 用錨點定位的行本來就會低。
        flag = ''
        if r['source'] == 'interp':
            flag = ' <- 插值(無音訊證據)'
            low.append(i)
        elif r['source'] == 'moved':
            flag = ' <- 錨點落在間奏,已重新安置'
            low.append(i)
        elif rate > 9.0:
            flag = ' <- 時長偏短?'
            low.append(i)
        elif nchars >= 6 and rate < 1.0:
            flag = ' <- 時長偏長?'
            low.append(i)
        print(f"  L{i + 1:02d} {r['time']:7.2f} {r['end']:7.2f} {dur:6.2f} "
              f"{rate:6.1f} {str(r['source']):>6} {gap:6.2f}{flag}", flush=True)
        prev_end = r['end']
    print("  " + "-" * 78, flush=True)
    if low:
        print(f"  [align] 建議人工檢查的行: {[i + 1 for i in low]}", flush=True)
    else:
        print("  [align] 每行都有音訊證據，時長也都在合理範圍", flush=True)
