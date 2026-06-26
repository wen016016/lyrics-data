\# 🎵 歌詞收藏站



一個收錄各類歌曲歌詞的靜態網頁，支援 YouTube 同步播放、歌詞高亮跟隨、振假名（ふりがな）顯示等功能。


\## ✨ 功能特色



\- \*\*歌手篩選\*\* — 按歌手分類快速找到想看的歌曲

\- \*\*分頁切換\*\* — Tab 式切換不同歌曲

\- \*\*YouTube 同步播放\*\* — 內嵌 YouTube 播放器，歌詞隨播放進度自動高亮

\- \*\*點擊跳轉\*\* — 點擊任一句歌詞即可跳轉到對應時間點

\- \*\*振假名標註\*\* — 使用 `<ruby>` 標籤顯示漢字讀音

\- \*\*自動捲動\*\* — 播放中歌詞自動捲動至畫面中央

\- \*\*依歌手排序\*\* — 全部歌曲依歌手名稱分組顯示



\## 📂 專案結構

├── index.html      # 主頁面（HTML + CSS + JavaScript）

├── songs.json      # 歌曲資料（歌詞、時間軸、影片 ID 等） 

└── README.md       # 本文件



\## 🚀 啟動方式



\### 線上版



直接前往：\*\*https://mylovesong.netlify.app/\*\*



\### 本地版



1\. 在專案資料夾開啟 CMD

2\. 執行以下指令啟動本地伺服器：python -m http.server 8000

3\. 開啟瀏覽器前往：\*\*http://localhost:8000/\*\*



> ⚠️ \*\*注意：\*\* 使用本地版需將 `index.html` 中的

> fetch('https://wen016016.github.io/lyrics-data/songs.json?\_=' + new Date().getTime())


> 改為


> fetch('songs.json')




\## 📝 新增歌曲



在 `songs.json` 的陣列中新增一筆物件即可，格式如下：

{ "title": "歌曲名稱", 

&#x20; "artist": "歌手名稱", 

&#x20; "credits": { "lyrics": "作詞者", "music": "作曲者" }, 

&#x20; "color": "#主題色hex", 

&#x20; "videoId": "YouTube影片ID", 

&#x20; "lines": \[ { "time": 0, "text": "第一句歌詞" }, { "time": 5.5, "text": "第二句歌詞" } ] }





\### 欄位說明



| 欄位 | 說明 |

|------|------|

| `title` | 歌曲標題 |

| `artist` | 歌手／團體名稱（用於篩選分類） |

| `credits.lyrics` | 作詞者（無則填 `null`） |

| `credits.music` | 作曲者（無則填 `null`） |

| `color` | 該曲目的主題色（HEX 格式） |

| `videoId` | YouTube 影片 ID（網址 `v=` 後面的部分，無則填 `null`） |

| `lines\[].time` | 該句歌詞的起始秒數 |

| `lines\[].text` | 歌詞內容，支援 `<ruby>漢字<rt>讀音</rt></ruby>` 格式 |



\## ⚠️ 注意事項



\- `songs.json` 必須是合法的 JSON 陣列，開頭為 `\[`，結尾為 `]`

\- 每首歌物件之間需以 `,` 分隔

\- 振假名使用 HTML `<ruby>` 標籤語法

\- 需要網路連線才能載入 YouTube 播放器與 Google Fonts



\## 🛠️ 技術棧



\- \*\*HTML5\*\* / \*\*CSS3\*\* / \*\*Vanilla JavaScript\*\*（無框架）

\- \*\*YouTube IFrame API\*\* — 影片播放與同步

\- \*\*Google Fonts\*\* — Noto Serif JP、Cormorant Garamond



\## 📄 授權



歌詞版權歸原作者及相關權利人所有，本專案僅供個人學習與收藏用途。

