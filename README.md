# 🎵 歌詞收藏站

一個收錄日文歌曲歌詞的靜態網頁，支援 YouTube 同步播放、歌詞高亮跟隨、振假名（ふりがな）顯示、中文翻譯對照等功能。

## ✨ 功能特色

- **歌手篩選** — 按歌手分類快速找到想看的歌曲
- **分頁切換** — Tab 式切換不同歌曲
- **YouTube 同步播放** — 內嵌 YouTube 播放器，歌詞隨播放進度自動高亮
- **點擊跳轉** — 點擊任一句歌詞即可跳轉到對應時間點
- **振假名標註** — 使用 `<ruby>` 標籤顯示漢字讀音
- **自動捲動** — 播放中歌詞自動捲動至畫面中央
- **中文翻譯對照** — 支援日文 / 中文 / 全部顯示模式切換
- **播放速度控制** — 0.5x / 0.75x / 1x / 1.25x
- **單句循環 / A-B 段落循環** — 練歌好幫手
- **時間偏移微調** — ±0.1 秒步進即時調整歌詞同步
- **深色主題** — 護眼閱讀

## 📂 專案結構

```
song/
├── index.html          # 歌詞播放器主頁面
├── add_song.html       # 新增歌曲的 GUI 表單
├── songs.json          # 歌曲資料庫（歌詞、時間軸、假名、翻譯）
├── server.py           # 本地伺服器（靜態檔案 + 歌曲處理 API）
├── README.md
└── tools/
    ├── add_song_new.py     # 主流程：下載音檔 → 對齊 → 標假名 → 翻譯 → 寫入
    ├── aligner.py          # 時間軸對齊引擎（全域 CTC forced alignment）
    ├── setup_whisperx.ps1  # 一鍵安裝環境腳本
    ├── requirements.txt    # Python 相依套件
    └── temp_audio/         # 暫存音檔
```

## 🚀 啟動方式

### 線上版（僅播放器，無法新增歌曲）

直接前往：**https://mylovesong.netlify.app/**

### 本地版（完整功能）

> ⚠️ 首次使用需要先完成下方「環境安裝」步驟。

1. 在專案資料夾按 `Shift + 右鍵` → 選「在這裡開啟 PowerShell」
2. 輸入以下指令啟動伺服器：
   ```powershell
   .\.venv-whisperx\Scripts\python.exe server.py
   ```
3. 看到 `伺服器啟動於 http://localhost:8000/` 就代表成功了
4. 開啟瀏覽器前往：
   - **http://localhost:8000/** → 歌詞播放器
   - **http://localhost:8000/add_song.html** → 新增歌曲
5. 要關閉伺服器，回到終端機按 `Ctrl + C`

> ⚠️ **注意：** 如果是部署到 GitHub Pages 等靜態站台，需將 `index.html` 中的：
> ```js
> fetch('https://wen016016.github.io/lyrics-data/songs.json?_=' + new Date().getTime())
> ```
> 改為本地版：
> ```js
> fetch('songs.json')
> ```

## 🔧 環境安裝（首次使用）

需要：**Python 3.12**（3.13+ 不支援）、**FFmpeg**

### 快速安裝

在專案資料夾開啟 PowerShell，執行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\tools\setup_whisperx.ps1
```

### 手動安裝

如果快速安裝失敗，可以手動安裝：

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg.Shared
```

安裝完**重開終端機**，然後建立虛擬環境：

```powershell
py -3.12 -m venv .venv-whisperx
.\.venv-whisperx\Scripts\python.exe -m pip install --upgrade pip
.\.venv-whisperx\Scripts\python.exe -m pip install -r .\tools\requirements.txt
```

`requirements.txt` 包含：

| 套件 | 用途 | 必要性 |
|------|------|--------|
| `yt-dlp` | 下載 YouTube 音檔 | 必要 |
| `torch` `torchaudio` `transformers` | wav2vec2 CTC 對齊 | 必要 |
| `stable-ts` `faster-whisper` | Whisper 轉錄（時間軸錨點） | 必要 |
| `pykakasi` | 漢字→假名（振假名與日文對齊） | 必要 |
| `demucs` | 人聲分離 | 選用，但強烈建議 |
| `opencc-python-reimplemented` | 繁簡轉換 | 中文歌必要 |
| `anthropic` 或 `openai` | 自動翻譯 | 選用 |

### 自動翻譯（選用）

設定任一組金鑰就會自動翻譯，不必額外跑一次流程：

```powershell
.\.venv-whisperx\Scripts\python.exe -m pip install anthropic
$env:ANTHROPIC_API_KEY = "你的API金鑰"
```

或用 OpenAI：

```powershell
.\.venv-whisperx\Scripts\python.exe -m pip install openai
$env:OPENAI_API_KEY = "你的API金鑰"
```

兩個都有設定時優先使用 Claude。沒設定金鑰時會跳過翻譯並提示，
時間軸與振假名仍然照跑，之後可用 `--import-zh` 補上翻譯。

## 📝 新增歌曲

### 方法一：使用 GUI 表單（推薦，全自動）

1. 啟動伺服器後，瀏覽器打開 **http://localhost:8000/add_song.html**
2. 依步驟填入 YouTube 連結、歌名、歌手、歌詞、翻譯
3. 點擊「🚀 開始處理」，系統會自動完成所有步驟：
   - 下載 YouTube 音檔
   - 人聲分離（約 5 分鐘）
   - 時間軸對齊（約 10 分鐘）
   - 標注振假名
   - 翻譯成中文（沒填中文歌詞時自動翻譯，需設定 API 金鑰）
   - 寫入 `songs.json`
4. 完成後重新整理播放器頁面即可看到新歌曲

> 一首 5 分鐘的歌在 CPU 上約需 20-25 分鐘。有 NVIDIA GPU 可用 `--device cuda` 大幅加速。

> 全程不需要使用終端機或輸入任何指令！

### 方法二：使用指令

```powershell
.\.venv-whisperx\Scripts\python.exe .\tools\add_song_new.py --add --title "歌名" --artist "歌手" --video-id "影片ID" --lyrics .\lyrics.txt
```

歌詞檔（`lyrics.txt`）格式為一行一句純文字，不需要加時間標記。

#### 可選參數

| 參數 | 說明 |
|------|------|
| `--lyrics-by "作詞者"` | 作詞者 |
| `--music-by "作曲者"` | 作曲者 |
| `--color "#HEX"` | 主題色（預設 `#c9a96e`） |
| `--zh-file 檔案` | 直接附上中文歌詞（行數需與歌詞相同）；沒給的行才自動翻譯 |
| `--no-translate` | 完全跳過翻譯 |
| `--translate-model` | 翻譯模型（預設 Claude 用 `claude-opus-5`、OpenAI 用 `gpt-4o`） |
| `--device cuda` | 使用 GPU 加速（需 NVIDIA + CUDA） |
| `--lang ja` | 指定歌詞語言（預設從歌詞的文字系統自動判斷） |
| `--align-model NAME` | 指定 HuggingFace CTC 對齊模型，覆蓋語言預設 |
| `--no-separate` | 跳過人聲分離（快約 5 分鐘，但有合音／厚伴奏的歌會變差） |
| `--no-anchor` | 跳過 Whisper 錨點（快很多，但長句可能漂移） |
| `--whisper-model` | 錨點用的 Whisper 模型（預設 `large-v3`） |

### 特殊念法：`表記{讀音}`

有些歌詞的讀音推不出來 —— 熟字訓、義訓、歌手自創讀音、日文歌裡的英文段落。
這種情況直接在歌詞檔裡標讀音，用半角大括號：

```
365{エブリデイ}
run{ラン}
嗤う{わらう}
```

| 效果 | 說明 |
|------|------|
| 畫面顯示 | `365`、`run`、`嗤う`（大括號不會顯示） |
| 時間軸對齊 | 用 `エブリデイ`、`ラン`、`わらう` 去比對音訊 |
| 振假名 | 直接採用你標的讀音，不讓 pykakasi 蓋掉 |

**什麼時候需要標：**

- 數字／英文有特殊唸法（`365` 唱成 everyday）
- 日文歌裡夾英文段落 —— 不標的話英文字母不在日文對齊模型的字表裡會被丟掉
- 歌手用了非字典讀音

標註會存進 `songs.json` 的 `src` 欄位，重跑流程時不會遺失。

### 合唱／和聲：行首 `+`

兩句同時唱（主唱 + 和聲、對唱）時，在第二句行首加 `+`，播放時兩句會一起亮：

```
主旋律這一句
+和聲同時唱的另一句
```

連續多行都加 `+` 就會併成同一組，整組共用同一個時間範圍。

**為什麼要手動標，不能自動偵測：** 對齊的兩個機制（全域 CTC Viterbi、Whisper
錨點 + 序列比對）都建立在「歌聲是單一條單調序列」這個前提上，結構上不可能吐出
「同一時間有兩句」的結果。要自動偵測得先把主唱和和聲分離成兩軌，那是另一個
層級的問題。你聽得出來哪幾句是疊在一起的，直接標最可靠。

合併時取兩句的**聯集**而不是各給一半 —— 兩個聲部同時唱時 ASR 看到的是交錯的
單一串流，對齊只能把兩句的 token 前後排開，各自的一半都是錯的，但聯集是對的。

`+` 也會存進 `src` 欄位。可以和讀音標註併用：`+365{エブリデイ}`。

### 時間軸對齊怎麼運作

歌詞是已知的真值，所以走的是 **forced alignment**，不是「先辨識再猜哪句」：

```
音訊
 -> demucs 抽人聲                   伴奏與合音不再干擾對齊
 -> Whisper large-v3 轉錄           骨架：哪句大概在哪一段
 -> 全域序列比對 (Needleman-Wunsch)  歌詞字元 <-> ASR 字元，無門檻、天生單調
 -> wav2vec2 frame-level emission
 -> 人聲活動遮罩                     間奏在哪，由 emission 自己判斷
 -> 全域 CTC Viterbi + 位置先驗       整首歌一次算完，精確到 20ms
 -> 能量延伸行尾                     補回拖長音，字幕不會唱到一半就熄
```

關鍵設計：**每一行的 `time` 和 `end` 都來自音訊 frame，不從字數推算。**
用字數 ÷ 每秒字數去合成時長，會讓時間軸變成字數節拍器，
拖長音的句子一定提早熄掉，誤差還會沿著整首歌往後傳遞。

處理完會印出每行的 start / end / 時長 / 字每秒 / 邊界來源，時長異常或缺音訊證據的行
會被標記，方便直接鎖定需要人工檢查的句子。`來源` 欄位的意思：

| 來源 | 意思 |
|------|------|
| `anchor` | 邊界來自 Whisper 錨點（日文歌通常是這個） |
| `ctc` | 邊界來自 wav2vec2 聲學對齊（拉丁語系歌曲通常是這個） |
| `moved` | 錨點不可靠，已依人聲區間重新安置 —— 建議人工確認。兩種情況會觸發：錨點整段落在間奏裡；或前面有沒人認領的人聲、而這句被擠壓成不可能唱完的長度 |
| `interp` | 完全沒有音訊證據，由前後行插值 —— 一定要人工確認 |

### 已知限制

**合音／和聲。** demucs 分離出的人聲軌會包含主唱和所有和聲。
和聲唱同一句歌詞時沒有影響（音量更大反而更好對）；
和聲唱不同內容（對唱、ad-lib、backing）時，那些字會變成 ASR 的多餘輸出，
由全域序列比對當成插入吸收掉 —— 這是全域比對本來就擅長的事，通常不會造成錯位。
真的出錯時會被標成 `moved` 或時長異常，直接看報表就能鎖定。

**很吵或人聲很悶的混音。** 人聲分離的品質決定上限，分離不乾淨時精度會下降。

**沒有人聲分離時。** `--no-separate` 會讓「間奏偵測」只能靠音素密度，
拖長音有機率被誤判成間奏。有合音或厚伴奏的歌不建議關掉。

### 中文翻譯

有設定 API 金鑰時，處理歌曲會自動翻譯，不需要另外跑流程。行為是：

- `--zh-file` 給過的行 → 用你的翻譯
- 其餘的行 → 自動翻譯
- 完全不想翻 → `--no-translate`

翻譯是分段送出（每 25 行）並用編號回填，所以模型多一行少一行也不會讓整首歌
錯位一格；缺的行會明確列出來讓你補。

事後補翻譯：

```powershell
.\.venv-whisperx\Scripts\python.exe .\tools\add_song_new.py --import-zh "歌名" --zh-file .\translation.txt
```

翻譯檔格式為一行一句，順序對應歌詞（行首的 `1.` `1|` 編號會自動去掉）。

## 📋 songs.json 格式

```json
{
  "title": "歌曲名稱",
  "artist": "歌手名稱",
  "credits": {
    "lyrics": "作詞者",
    "music": "作曲者"
  },
  "color": "#HEX主題色",
  "videoId": "YouTube影片ID",
  "lines": [
    {
      "time": 3.02,
      "end": 9.7,
      "text": "歌詞（含 ruby 假名標記）",
      "plain": "歌詞（純文字）",
      "zh": "中文翻譯",
      "src": "帶讀音標註的原始歌詞（只有用到標註的行才有）"
    }
  ]
}
```

### 欄位說明

| 欄位 | 說明 |
|------|------|
| `title` | 歌曲標題 |
| `artist` | 歌手／團體名稱（用於篩選分類） |
| `credits.lyrics` | 作詞者（無則填 `null`） |
| `credits.music` | 作曲者（無則填 `null`） |
| `color` | 該曲目的主題色（HEX 格式） |
| `videoId` | YouTube 影片 ID（網址 `v=` 後面的部分，無則填 `null`） |
| `lines[].time` | 該句的起始秒數（自動產生，來自音訊 frame） |
| `lines[].end` | 該句的結束秒數（自動產生，來自音訊 frame）。播放器只在 `time ≤ 現在 < end` 期間高亮這句 |
| `lines[].text` | 歌詞內容，支援 `<ruby>漢字<rt>讀音</rt></ruby>` 格式 |
| `lines[].plain` | 歌詞純文字（不含 HTML） |
| `lines[].zh` | 中文翻譯（選填） |
| `lines[].src` | 帶 `表記{讀音}` 標註的原始歌詞，重跑流程時用來還原標註（只有用到標註的行才有） |

## ⚠️ 注意事項

- `songs.json` 必須是合法的 JSON 陣列，開頭為 `[`，結尾為 `]`
- 每首歌物件之間需以 `,` 分隔
- 振假名使用 HTML `<ruby>` 標籤語法
- 需要網路連線才能載入 YouTube 播放器與 Google Fonts
- YouTube 可能會擋 yt-dlp 下載（403），此時請手動下載音檔轉成 WAV 放到 `tools/temp_audio/影片ID.wav`
- 一首 5 分鐘的歌在 CPU 上約需 20-25 分鐘（人聲分離 ~5 分、Whisper ~10 分、其餘 ~5 分）。
  有 NVIDIA GPU 可用 `--device cuda` 大幅加速；趕時間可用 `--no-separate` 或 `--no-anchor`
- 對齊失敗時會直接報錯，不會寫入猜測的時間軸。看報表的 `來源` 欄位判斷哪些行要人工確認
- 播放時可用 Offset（±0.1 秒）即時微調整首歌的偏移
- 首次執行會自動下載模型（demucs ~80MB、wav2vec2 ~1.2GB、Whisper large-v3 ~3GB）

## 🛠️ 技術棧

### 前端

- **HTML5** / **CSS3** / **Vanilla JavaScript**（無框架）
- **YouTube IFrame API** — 影片播放與同步
- **Google Fonts** — Noto Serif JP、Cormorant Garamond

### 時間軸對齊

- **demucs** — 人聲分離
- **faster-whisper** + **stable-ts** — 轉錄，提供「哪句在哪一段」的錨點
- **wav2vec2 (XLSR-53)** — frame-level CTC posterior，提供 20ms 解析度的邊界
- **Needleman-Wunsch** — 歌詞與 ASR 的全域序列比對
- **CTC Viterbi**（自行實作於 `tools/aligner.py`）— 全曲一次性強制對齊

### 文字處理

- **pykakasi** — 漢字假名轉換、振假名
- **opencc** — 繁簡轉換（中文歌對齊用）
- **Claude** 或 **OpenAI** — 中文翻譯（選用）

## 📄 授權

歌詞版權歸原作者及相關權利人所有，本專案僅供個人學習與收藏用途。
