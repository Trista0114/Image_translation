# 圖片文字翻譯服務(Image Translation Service)
本專案旨於熟悉AI native coding，全部作業內容使用claude code執行，目標Image為雜誌封面或是帶有公式的文獻內容。

全程**不需要任何 API key**:OCR 在本地執行(PaddleOCR),翻譯使用 deep-translator 的 Google 免費端點(需要網路連線,但免金鑰)，無使用任何模型因此輸出結果不甚完美(尤其需要高判斷力的雜誌封面)，條件允許可以設計使用OCR+VLM或是使用可插拔設計(Gemini/Claude Api)來完善結果內容。

使用方式: 上傳一張含有文字的圖片並指定目標語言,服務會辨識圖片中的所有文字、抹除原文,並將譯文以貼近原本樣式(位置、顏色、粗細)的方式重繪回圖片上。


## 運作流程

整個 pipeline 分為四個步驟(可透過網頁 UI 或 `POST /steps` 檢視每個中間階段):

1. **偵測文字** — 使用 PaddleOCR 在本地偵測每行文字與其邊界框(bounding box),並過濾誤偵測(信心分數門檻、重複框去重)。
2. **抹除原文** — 依背景類型選擇策略:純色色塊直接重建、平滑漸層逐列內插重建、照片紋理才使用 OpenCV inpainting(僅遮筆畫,避免暈染),並有殘影檢查重修機制。
3. **翻譯** — 使用 deep-translator 的 `GoogleTranslator` 翻譯文字;會先將同一段落的行合併後再翻譯,讓譯文有完整的句子上下文。
4. **重繪譯文** — 使用 Pillow 將譯文貼回原本的框內:自動以二分搜尋找出能塞進框內的最大字級、CJK/拉丁混排斷行,並保留原文的顏色與粗細。

## 特色

- **免 API key、免付費服務** — 本地 OCR + 免費翻譯端點。
- **段落分組翻譯** — 合併屬於同一段落的 OCR 行,提升翻譯品質;大標題與小字副標依字級差自動分開,不會被擠成一團。
- **數學公式保護** — 自動偵測看起來像公式/方程式的行,完整保留不翻譯、不抹除;被 OCR 拆開的公式碎片(如分子、式號)會被吸收進公式一併保護(overlay 中以綠框標示)。
- **品牌刊名保護** — 雜誌刊名(VOGUE、PSYCHOLOGIES…)等頂部大型品牌字自動辨識並原樣保留,不翻譯、不重繪(overlay 中以藍框標示,一般文字為紅框)。
- **背景重建** — 文字底下是純色色塊或平滑漸層時,直接重建背景而非 inpaint,避免大面積修復的塊狀暈染。
- **樣式保留** — 取樣原文顏色、估計粗體字重,重繪時融合到局部背景中;字色以色彩距離(而非亮度差)判斷易讀性,深色底上的紅色標題不會被誤改成白字。
- **OCR 誤偵測過濾** — 依信心分數過濾照片紋理誤判(嘴唇、飾品高光),並對同一行字的重複偵測去重。
- **跨平台字型** — 自動在 Windows / Linux / macOS 尋找可用的 CJK 與拉丁字型(微軟正黑體、Noto CJK、PingFang、Arial、DejaVu Sans 等)。
- **可選 GPU** — 透過環境變數 `OCR_GPU` 啟用。

## 技術棧

| 項目 | 使用技術 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| OCR | PaddleOCR / PaddlePaddle(本地執行) |
| 翻譯 | deep-translator(GoogleTranslator 免費端點) |
| 影像處理 | Pillow、OpenCV(headless)、NumPy |

需求:Python 3.9 以上。

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
uvicorn translate_api:app --reload --port 8000
```

啟動後開啟 <http://localhost:8000/> 即可使用內建的網頁測試介面(上傳圖片、選擇語言、檢視四個步驟的中間結果)。

> **注意**
> - 首次執行時 PaddleOCR 會自動下載偵測/辨識模型,需要一些時間。
> - 翻譯步驟需要網路連線(使用 Google 公開端點),但不需要 API key。

## API 端點

### `GET /`

內建 HTML 測試頁。

### `GET /health`

健康檢查,回傳引擎與 GPU 狀態:

```json
{"status": "ok", "engine": "paddleocr", "gpu": false}
```

### `POST /translate`

翻譯圖片,以 `multipart/form-data` 上傳:

| 欄位 | 必填 | 說明 |
|---|---|---|
| `image` | ✅ | 圖片檔案 |
| `target_language` | ✅ | 目標語言,例如 `繁體中文`、`english`、`ja`(見下方支援語言) |
| `source_languages` | | 來源語言,逗號分隔(如 `en,fr`);留空 = 自動偵測 |
| `response_format` | | `image`(預設,回傳 PNG)或 `json`(回傳 base64 圖片 + 各區域的原文/譯文) |

範例:

```bash
# 回傳翻譯後的 PNG 圖片
curl -X POST http://localhost:8000/translate \
  -F "image=@test_image/Cover_1.png" \
  -F "target_language=繁體中文" \
  -o translated.png

# 回傳 JSON(含 base64 圖片與每個文字區域的翻譯結果)
curl -X POST http://localhost:8000/translate \
  -F "image=@test_image/image5.jpg" \
  -F "target_language=english" \
  -F "response_format=json"
```

### `POST /steps`

與 `/translate` 輸入相同(`image`、`target_language`、`source_languages`),回傳 pipeline 每個中間階段的 base64 PNG 與各區域的翻譯結果,供逐步檢視(網頁 UI 即使用此端點)。

## 支援語言

繁體中文、簡體中文、英文、日文、韓文、法文、德文、西班牙文。

`target_language` 接受多種寫法(中文名稱、英文名稱或語言代碼),例如 `繁體中文` / `traditional chinese` / `zh-TW` 均可。來源語言會對應到 PaddleOCR 專用的辨識模型(`ch`、`chinese_cht`、`japan`、`korean`、`fr`、`de`、`es`、`en`)。

## 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `OCR_GPU` | `0` | 設為 `1` 以使用 GPU 執行 OCR;預設使用 CPU |
| `PADDLE_PDX_CACHE_HOME` | Windows 上為 `C:\paddlex_cache` | PaddleX 模型快取路徑。Windows 上預設改導向純 ASCII 路徑,避免使用者資料夾含非 ASCII 字元(如中文使用者名稱)時 PaddleX 讀取模型失敗 |

## 測試

使用 `test_image/` 內的範例圖片(雜誌封面 `Cover_*`、含公式論文 `formula_*`),透過網頁 UI 或上方的 curl 範例手動驗證;`Result/` 為代表性輸出結果,`test/` 為開發期間各輪迭代的測試截圖。

## 專案結構

```
pega_assign/
├── translate_api.py   # 完整服務(OCR、翻譯、重繪、API、網頁 UI)
├── requirements.txt   # Python 依賴
├── test_image/        # 測試用來源圖片(雜誌封面、含公式論文)
├── Result/            # 代表性翻譯結果(四步驟視覺化)
└── test/              # 開發迭代過程的測試截圖
```
