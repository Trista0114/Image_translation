"""
Image translation service — local OCR version (no API key).

Input : an image + a target language
Output: the same image with all visible text translated into the target
        language, re-rendered in place.

Pipeline
--------
1. OCR with PaddleOCR (runs locally): detects text lines and their bounding
   boxes. Higher recall/accuracy than EasyOCR, especially for dense or
   non-English text.
2. Translate each line with deep-translator's GoogleTranslator (free web
   endpoint, no API key).
3. Re-render with Pillow: inpaint away the original text, then draw the
   translated text fitted into the original box.

Run:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/ for a small test page.

Notes
-----
* PaddleOCR downloads its detection/recognition models on first use.
* Translation needs internet access (Google's public endpoint) but no API key.
  For a fully offline setup, swap GoogleTranslator for Argos Translate.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

# These flags must be set BEFORE importing paddleocr/paddle. They avoid a
# PaddlePaddle CPU oneDNN/PIR compatibility issue where newer PP-OCRv6 model
# files (the "inference.json" PIR format) fail to load with a misleading
# "Cannot open file ... inference.json" error on some paddlepaddle builds.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# On Windows, PaddleX's underlying C++ file-open check can fail to find model
# files when the cache path contains non-ASCII characters (e.g. a Chinese
# Windows username like C:\Users\陳芃\.paddlex\...), even though the file is
# physically present. Redirect the cache to a plain ASCII path to avoid this.
# (Set PADDLE_PDX_CACHE_HOME yourself beforehand if you want a different spot.)
if os.name == "nt":
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", r"C:\paddlex_cache")

from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

USE_GPU = os.environ.get("OCR_GPU", "0") not in ("0", "", "false", "False")


def _first_existing(*candidates: str) -> str:
    """Return the first path that exists on this machine; else the first
    candidate (so callers get a sane default / clear error if truly missing)."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


# Each of these tries Windows fonts first, then Linux/macOS fonts, so the
# same script works across operating systems without editing paths by hand.
CJK_FONT = _first_existing(
    "C:/Windows/Fonts/msjh.ttc",       # Windows: Microsoft JhengHei (Traditional)
    "C:/Windows/Fonts/msyh.ttc",       # Windows: Microsoft YaHei (Simplified)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",  # macOS
)
CJK_FONT_BOLD = _first_existing(
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    CJK_FONT,
)
LATIN_FONT = _first_existing(
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)
LATIN_FONT_BOLD = _first_existing(
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    LATIN_FONT,
)

CJK_TARGET_CODES = ("zh-cn", "zh-tw", "zh", "ja", "ko")

# Friendly names / aliases -> Google Translate language codes.
LANG_ALIASES = {
    "繁體中文": "zh-TW", "繁体中文": "zh-TW", "traditional chinese": "zh-TW",
    "简体中文": "zh-CN", "簡體中文": "zh-CN", "simplified chinese": "zh-CN",
    "中文": "zh-CN", "chinese": "zh-CN",
    "日本語": "ja", "日文": "ja", "japanese": "ja",
    "한국어": "ko", "韓文": "ko", "韩文": "ko", "korean": "ko",
    "english": "en", "英文": "en", "英語": "en",
    "français": "fr", "法文": "fr", "french": "fr",
    "deutsch": "de", "德文": "de", "german": "de",
    "español": "es", "西班牙文": "es", "spanish": "es",
}

app = FastAPI(title="Image Translation Service (local OCR, auto-detect source)")


# --------------------------------------------------------------------------- #
# OCR + translation
# --------------------------------------------------------------------------- #

@dataclass
class Region:
    box: tuple[int, int, int, int]  # pixel coords x0,y0,x1,y1
    original: str
    translation: str
    is_formula: bool = False  # True → leave this region's pixels untouched
                               # (no inpaint, no redraw); our renderer can't
                               # typeset subscripts/superscripts, so the
                               # safest treatment for a real formula is to
                               # not touch it at all.
    keep: bool = False         # True → brand/logo (e.g. a magazine masthead)
                               # we deliberately don't translate; treated like
                               # a formula for pixels — kept exactly as-is so
                               # the stylised original logo is preserved.


def _untouched(r: "Region") -> bool:
    """Region whose original pixels must be preserved (formula or kept logo)."""
    return r.is_formula or r.keep


# Map our source-language codes to a PaddleOCR recognition model. PaddleOCR
# ships one model per language family; we collapse the comma list to one model.
# fr/de/es get their own dedicated (accent-aware) models; anything else we
# don't recognise (or "auto-detect") falls back to the plain "en" model,
# which only reads unaccented Latin letters.
def _paddle_lang(source_langs: tuple[str, ...]) -> str:
    for s in (s.strip().lower() for s in source_langs if s.strip()):
        if s.startswith("zh-tw") or s.startswith("cht"):
            return "chinese_cht"
        if s.startswith("ch") or s.startswith("zh"):
            return "ch"
        if s.startswith("ja"):
            return "japan"
        if s.startswith("ko"):
            return "korean"
        # PP-OCRv5/v6 ship dedicated recognition models for these Latin-script
        # languages, with accented characters (é, è, ê, ç, ñ, ü, …) in their
        # dictionaries. Falling back to the plain "en" model here was the bug:
        # it can't read accents at all, so it silently misreads real letters,
        # which then feeds garbled text into translation.
        if s.startswith("fr"):
            return "fr"
        if s.startswith("de"):
            return "de"
        if s.startswith("es"):
            return "es"
    return "en"


# Map our source-language selection to a Google Translate source code.
# Returns "auto" when nothing was selected, letting Google guess the
# language — auto-detect can misfire on short, out-of-context snippets
# (e.g. a single legal term with no surrounding sentence), so picking a
# specific language here avoids that failure mode when the user knows it.
def _google_source_lang(source_langs: tuple[str, ...]) -> str:
    for s in (s.strip().lower() for s in source_langs if s.strip()):
        if s.startswith("zh-tw") or s.startswith("cht"):
            return "zh-TW"
        if s.startswith("ch") or s.startswith("zh"):
            return "zh-CN"
        if s.startswith("ja"):
            return "ja"
        if s.startswith("ko"):
            return "ko"
        if s.startswith("fr"):
            return "fr"
        if s.startswith("de"):
            return "de"
        if s.startswith("es"):
            return "es"
        if s.startswith("en"):
            return "en"
    return "auto"


@lru_cache(maxsize=8)
def get_reader(lang: str) -> PaddleOCR:
    """Build (and cache) a PaddleOCR reader for a recognition language.

    Detection is tuned for higher recall: a lower box threshold and a larger
    unclip ratio keep fainter / smaller text, and a bigger detection side length
    preserves small glyphs.
    """
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,  # avoids a oneDNN crash on some CPU builds
        device="gpu" if USE_GPU else "cpu",
        text_det_thresh=0.2,           # default 0.3 — keep fainter pixels
        text_det_box_thresh=0.4,       # default 0.6 — keep lower-confidence boxes
        text_det_unclip_ratio=1.8,     # default 1.5 — grow boxes a little
        text_det_limit_side_len=1536,  # detect on a larger canvas
        text_det_limit_type="max",
    )


def normalize_target(target_language: str) -> str:
    key = target_language.strip().lower()
    if key in (v.lower() for v in LANG_ALIASES.values()):
        return target_language.strip()
    return LANG_ALIASES.get(target_language.strip(), LANG_ALIASES.get(key, target_language.strip()))


# --------------------------------------------------------------------------- #
# Formula detection + paragraph grouping
# --------------------------------------------------------------------------- #

_FORMULA_FUNC_RE = re.compile(r"\b(max|min|sin|cos|tan|log|exp|argmax|argmin|softmax)\b", re.I)
_SUBSCRIPT_VAR_RE = re.compile(r"\b[A-Za-z]{1,3}\d\b")           # W1, b2, d1 …
_EQUATION_LABEL_RE = re.compile(r"^\(?\d{1,3}\)?$")               # "(2)", "12"


def _looks_like_formula(text: str) -> bool:
    """Heuristic: does this OCR line look like a math formula/equation
    rather than a natural-language sentence?

    Used to (a) keep formula lines out of paragraph merging, and (b) skip
    translating + redrawing them entirely — our renderer can't typeset
    subscripts/superscripts/italics, so the safest treatment for a real
    formula is to leave the original pixels completely untouched.
    """
    t = text.strip()
    if not t:
        return False
    if _EQUATION_LABEL_RE.match(t):            # bare equation number, e.g. "(2)"
        return True

    # Guard against false positives: a line with plenty of real words is a
    # sentence, even if it happens to contain one inline math symbol (e.g.
    # "divide each by √dk" or "we scale the dot products by 1/√dk in the
    # softmax"). A single embedded symbol/function name shouldn't get an
    # entire prose sentence excluded from translation — only lines that are
    # MOSTLY notation (few or no real words) should count as a formula.
    word_tokens = re.findall(r"[A-Za-z]{3,}", t)
    if len(word_tokens) >= 6:
        return False

    if re.search(r"[=<>≤≥≈∑∏∫√±×÷∂∇πθλμσ∞]", t):  # explicit math operators/symbols
        return True
    if _SUBSCRIPT_VAR_RE.search(t) and len(t) < 40:  # W1, b2, d_model style tokens
        return True
    if _FORMULA_FUNC_RE.search(t) and len(t.split()) <= 6:  # short line ~ "max(0, ...)"
        return True
    letters = sum(ch.isalpha() for ch in t)
    if letters == 0:                            # no letters at all → digits/symbols only
        return True
    # 高符號/數字比例 → 公式。但這條只適用於「幾乎沒有真實單字」的行:
    # 雜誌副標常見「2050's Power Icons: Style,」「Science: Your 2050」這種
    # 含年份與冒號的正常語句,數字+標點一多就會被誤判成公式而整行不翻譯。
    # 走到這裡代表前面沒驗出任何數學符號/下標記號,所以只要有 ≥3 個真實
    # 單字就視為語句;比例也改以非空白字元計,不讓空格灌水撐高分母。
    if len(word_tokens) >= 3:
        return False
    compact = re.sub(r"\s+", "", t)
    if len(compact) < 60 and (1 - letters / max(1, len(compact))) > 0.35:
        return True
    return False


_STRONG_MATH_RE = re.compile(r"[=<>≤≥≈∑∏∫√±×÷∂∇]")


def _absorb_formula_fragments(items: list[dict]) -> None:
    """把緊鄰公式的短小片段也歸類為公式（就地修改 is_formula 旗標）。

    置中的大型公式常被 OCR 拆成好幾個小框：分母（√dk）、括號、式號 (1)
    這些含數學符號的片段判得出是公式，但**純字母的分子（例如 QKT）**
    靠文字本身判斷不出來。它若被當一般文字處理，公式的分子會被 inpaint
    抹掉、再蓋上一塊翻譯文字，整條公式就毀了。規則：不含空白的短片段
    （≤12 字元），若緊貼著一個公式「錨點」，就視為公式的一部分（完整保
    留、不翻譯）。被吸收的片段也加入錨點集合，反覆直到穩定。

    兩道防呆，避免在雜誌封面上連鎖誤吸：
      * 錨點必須有**明確的數學證據**（=、√ 等運算子、函數名、下標變數）。
        `30+`、`#1`、`ISSN 2002-4401` 這類純數字符號的版面元素雖自身判為
        公式（保留它們沒錯），但不能當錨點——否則封面上緊鄰的大字
        （ELON、TECH…）會一個接一個被連鎖吸收成「公式」而整塊不翻譯。
      * 鄰接距離從「一個行高」收緊為「較小框高的 0.6 倍」。雜誌大字的行
        高極大，舊 margin 會把半個版面外的字都算成「緊鄰」。
    """
    def _strong(text: str) -> bool:
        return bool(_STRONG_MATH_RE.search(text) or _FORMULA_FUNC_RE.search(text)
                    or _SUBSCRIPT_VAR_RE.search(text))

    anchors = [it["box"] for it in items if it["is_formula"] and _strong(it["original"])]
    if not anchors:
        return
    changed = True
    while changed:
        changed = False
        for it in items:
            text = it["original"].strip()
            if it["is_formula"] or len(text) > 12 or " " in text:
                continue
            x0, y0, x1, y1 = it["box"]
            h = y1 - y0
            for fx0, fy0, fx1, fy1 in anchors:
                margin = max(2, int(0.6 * min(h, fy1 - fy0)))
                if (x0 - margin <= fx1 and fx0 - margin <= x1
                        and y0 - margin <= fy1 and fy0 - margin <= y1):
                    it["is_formula"] = True
                    anchors.append(it["box"])
                    changed = True
                    break


def _dedup_overlapping_lines(lines: list[dict]) -> None:
    """去掉 OCR 對同一行文字的重複偵測（就地修改 lines）。

    高召回的偵測設定（放大畫布、低門檻）偶爾會對同一行字回報兩個高度
    重疊的框：一個完整、一個殘缺（例如「THE #1 SECRET TO」與「RET TO」）。
    兩個都留著，殘缺版的碎句會混進段落，翻譯就變得支離破碎。

    規則：兩框交集面積佔較小框六成以上 → 視為同一行的重複，保留辨識
    信心較高者；信心相近（差 ≤ 0.05）時保留字數較多（較完整）的那個。
    """
    def _better(a: dict, b: dict) -> dict:
        if abs(a["conf"] - b["conf"]) > 0.05:
            return a if a["conf"] > b["conf"] else b
        return a if len(a["text"]) >= len(b["text"]) else b

    for i in range(len(lines)):
        if lines[i] is None:
            continue
        for j in range(i + 1, len(lines)):
            if lines[i] is None or lines[j] is None:
                continue
            ax0, ay0, ax1, ay1 = lines[i]["box"]
            bx0, by0, bx1, by1 = lines[j]["box"]
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(ay1, by1) - max(ay0, by0)
            if iw <= 0 or ih <= 0:
                continue
            min_area = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
            if iw * ih < 0.6 * max(1, min_area):
                continue
            winner = _better(lines[i], lines[j])
            if winner is lines[i]:
                lines[j] = None
            else:
                lines[i] = None
    lines[:] = [ln for ln in lines if ln is not None]


def _merge_overlapping_text(items: list[dict]) -> None:
    """合併互相重疊的非公式文字區域（就地修改 items）。

    段落中若嵌著比行高還高的行內公式（例如 1/√dk），OCR 會把同一段拆成
    數個「框互相重疊」的群組；各自翻譯、各自貼回，譯文就會疊在一起變成
    無法閱讀的一團。這裡把重疊面積超過較小框三成的文字框合併成一個區域：
    框取聯集、原文依閱讀順序串接後一起翻譯。
    """
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            a = items[i]
            if a is None or a["is_formula"]:
                continue
            for j in range(i + 1, len(items)):
                b = items[j]
                if b is None or b["is_formula"]:
                    continue
                ax0, ay0, ax1, ay1 = a["box"]
                bx0, by0, bx1, by1 = b["box"]
                iw = min(ax1, bx1) - max(ax0, bx0)
                ih = min(ay1, by1) - max(ay0, by0)
                if iw <= 0 or ih <= 0:
                    continue
                min_area = min((ax1 - ax0) * (ay1 - ay0),
                               (bx1 - bx0) * (by1 - by0))
                if iw * ih < 0.3 * max(1, min_area):
                    continue
                # 字級差很多的兩塊不合併：避免大標題（如「MODERN CLASSIC」）
                # 把底下的小字副標吸進來一起翻譯、被縮成一小團。此合併原意是
                # 接回「被行內公式切開的同一段」，那種情況兩塊「單行字級」相近
                # ——用單行高度（line_h）而非整框高度判斷，才不會把「三行段落
                # ＋一行段落」這種同字級、不同框高的正常情況也擋掉。
                alh = a.get("line_h") or (ay1 - ay0)
                blh = b.get("line_h") or (by1 - by0)
                if max(alh, blh) > 1.6 * max(1, min(alh, blh)):
                    continue
                first, second = (a, b) if (ay0, ax0) <= (by0, bx0) else (b, a)
                a = items[i] = {
                    "box": (min(ax0, bx0), min(ay0, by0),
                            max(ax1, bx1), max(ay1, by1)),
                    "original": f'{first["original"]} {second["original"]}',
                    "is_formula": False,
                    "line_h": max(alh, blh),
                }
                items[j] = None
                changed = True
    items[:] = [it for it in items if it is not None]


def _mark_mastheads(items: list[dict], img_w: int, img_h: int) -> None:
    """就地標記「品牌刊名／logo」區塊為 keep=True（不翻譯、原樣保留）。

    雜誌刊名（PSYCHOLOGIES、VOGUE…）屬於品牌名，慣例不翻譯；而且它是特製
    的美術字，重繪成一般字型只會更醜，所以最好整塊保留原始像素。

    判定條件刻意保守，避免誤傷該翻譯的一般標題（例如論文的章節標題）：
      * 位於畫面最上方（前 22%）
      * 字很大（框高 ≥ 畫面高 4.5%，或達到全圖最高文字行的八成）
      * 單一詞（不含空白）且以字母為主 → 章節標題多為多字詞，不會中標
      * 全大寫或首字母大寫（品牌名常見寫法）

    字母下限設 2 而非 3：風格化刊名常被人物/物件擋住一部分，OCR 只抓到
    殘存的兩個字母（例如 VOGUE 只偵測到「UE」）。這種碎片若不保護，會被
    inpaint 抹掉再用普通字型重畫，毀掉刊名。其餘條件（頂部、巨大、單詞、
    全大寫）已足夠避免誤把一般短字保護起來。
    """
    text_items = [it for it in items if not it["is_formula"]]
    if not text_items:
        return
    tallest = max((it["box"][3] - it["box"][1]) for it in text_items)
    for it in text_items:
        text = it["original"].strip()
        x0, y0, x1, y1 = it["box"]
        h = y1 - y0
        letters = sum(ch.isalpha() for ch in text)
        if (y0 < 0.22 * img_h
                and (h >= 0.045 * img_h or h >= 0.8 * tallest)
                and " " not in text
                and letters >= 2 and letters >= 0.7 * len(text)
                and (text.isupper() or text.istitle())):
            it["keep"] = True


def _plausible_text(text: str, conf: float) -> bool:
    """過濾 OCR 誤偵測。

    圖片紋理(嘴唇、飾品高光、布料摺痕)常被偵測成一小框、辨識出無意義的
    短字串——若不濾掉,它會被翻譯後貼回照片上(例如嘴唇上出現譯文)。
    真實文字的辨識分數通常 > 0.85,紋理誤判多落在 0.3–0.7,因此:
      * 整體信心 < 0.5 → 捨棄(原本的 0.2 過於寬鬆)
      * 極短字串(字母/數字 ≤ 3 個)幾乎涵蓋所有紋理誤判 → 要求更高信心
    """
    if conf < 0.5:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    if alnum <= 3 and conf < 0.75:
        return False
    return True


def _group_into_paragraphs(lines: list[dict]) -> list[list[dict]]:
    """Group OCR lines that form one flowing paragraph so they get
    translated together as a full sentence, instead of each fragment being
    translated in isolation (which is what breaks context — e.g. a sentence
    getting cut mid-word between two lines).

    Formula-like lines (see `_looks_like_formula`) are never merged — each
    stays in its own single-line group — so a formula can't get glued into
    a sentence and corrupt either the translation or the "leave untouched"
    formula handling.

    A line only joins the paragraph above it when ALL of these hold:
      * neither line is formula-like
      * the vertical gap to the previous line is small relative to the
        SMALLER of the two line heights (ordinary line spacing, not a gap
        between unrelated blocks/paragraphs). Normalising by the smaller
        height — not the average — matters: a large title averaged with a
        small subtitle inflates the tolerance so much that the deliberate
        whitespace band under a title reads as "normal line spacing" and the
        subtitle gets glued on.
      * the two lines' horizontal ranges overlap substantially (same
        column/block — stops an unrelated sidebar element from merging in)
      * the two lines have similar height (same font size — stops a large
        title merging with the smaller subtitle/body underneath it). Magazine
        titles are often only ~1.4–1.5× their subtitle, so the threshold has
        to be tight (a loose 1.6 lets "STYLE REPORT" swallow its subtitles).
    """
    ordered = sorted(lines, key=lambda l: (l["box"][1], l["box"][0]))  # reading order

    groups: list[list[dict]] = []
    for line in ordered:
        line["is_formula"] = _looks_like_formula(line["text"])

        if line["is_formula"] or not groups:
            groups.append([line])
            continue

        prev = groups[-1][-1]
        if prev.get("is_formula"):
            groups.append([line])
            continue

        x0, y0, x1, y1 = line["box"]
        px0, py0, px1, py1 = prev["box"]
        h, ph = (y1 - y0), (py1 - py0)
        min_h = max(1, min(h, ph))

        vertical_gap = y0 - py1
        height_ratio = max(h, ph) / min_h
        overlap = min(x1, px1) - max(x0, px0)
        overlap_ratio = overlap / max(1, min(x1 - x0, px1 - px0))

        same_paragraph = (
            vertical_gap < 0.6 * min_h
            and height_ratio < 1.4
            and overlap_ratio > 0.3
        )

        if same_paragraph:
            groups[-1].append(line)
        else:
            groups.append([line])

    return groups


def detect_and_translate(
    img: Image.Image, target_language: str, source_langs: tuple[str, ...]
) -> list[Region]:
    """OCR the image, group lines into paragraphs (formulas excluded), then
    translate each group and return regions in pixel coords."""
    import numpy as np

    reader = get_reader(_paddle_lang(source_langs))
    base = img.convert("RGB")

    # Upscale small images so thin / small text is easier to detect, then map
    # the boxes back to the original coordinate system.
    long_side = max(base.width, base.height)
    scale = min(3.0, 1600 / long_side) if long_side < 1600 else 1.0
    ocr_img = (base.resize((round(base.width * scale), round(base.height * scale)),
                           Image.LANCZOS) if scale != 1.0 else base)

    results = reader.predict(np.array(ocr_img))
    if not results:
        return []
    res = results[0]
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    polys = res.get("rec_polys")
    if polys is None:
        polys = res.get("dt_polys", [])

    # 1. Collect valid raw lines in pixel coords — no translation yet.
    raw_lines: list[dict] = []
    for text, conf, poly in zip(texts, scores, polys):
        text = (text or "").strip()
        if not text or not _plausible_text(text, conf):
            continue
        pts = np.asarray(poly) / scale
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        raw_lines.append({"text": text, "box": (x0, y0, x1, y1),
                          "conf": float(conf)})

    if not raw_lines:
        return []

    # 1b. 同一行字被重複偵測(一完整、一殘缺)時只留一個,殘缺碎句
    #     才不會混進段落把翻譯攪碎。
    _dedup_overlapping_lines(raw_lines)

    # 2. Group consecutive natural-language lines into paragraphs so they get
    #    translated with full sentence context; formula lines stay isolated.
    groups = _group_into_paragraphs(raw_lines)

    # 3. 先整理出每個 group 的框、原文與公式旗標（翻譯延後），這樣才能先做
    #    「公式碎片吸收」，避免把公式分子（如 QKT）當一般文字翻譯掉。
    pending: list[dict] = []
    for group in groups:
        xs0 = [ln["box"][0] for ln in group]
        ys0 = [ln["box"][1] for ln in group]
        xs1 = [ln["box"][2] for ln in group]
        ys1 = [ln["box"][3] for ln in group]
        line_hs = sorted(ln["box"][3] - ln["box"][1] for ln in group)
        pending.append({
            "box": (min(xs0), min(ys0), max(xs1), max(ys1)),
            "original": " ".join(ln["text"] for ln in group),
            "is_formula": group[0].get("is_formula", False),
            "keep": False,
            "line_h": line_hs[len(line_hs) // 2],   # 單行字級（中位數）
        })

    _absorb_formula_fragments(pending)
    _merge_overlapping_text(pending)
    _mark_mastheads(pending, base.width, base.height)

    target_code = normalize_target(target_language)
    translator = GoogleTranslator(source=_google_source_lang(source_langs), target=target_code)

    regions: list[Region] = []
    for p in pending:
        if p["is_formula"] or p.get("keep"):
            # Formulas and brand mastheads are left completely untouched
            # downstream: no translation, no inpaint, no redraw (see the
            # _untouched() handling in remove_text / paste_translations).
            regions.append(Region(box=p["box"], original=p["original"],
                                  translation=p["original"],
                                  is_formula=p["is_formula"], keep=p.get("keep", False)))
            continue

        try:
            translation = translator.translate(p["original"]) or p["original"]
        except Exception:  # noqa: BLE001 - fall back to original on failure
            translation = p["original"]
        regions.append(Region(box=p["box"], original=p["original"],
                              translation=translation.strip(), is_formula=False))

    return regions


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _pick_font_path(target_language: str, bold: bool = False) -> str:
    """Pick a font matching the target script and the original weight (bold?)."""
    code = normalize_target(target_language).lower()
    is_cjk = any(code.startswith(c) for c in CJK_TARGET_CODES)
    if is_cjk:
        cand = CJK_FONT_BOLD if bold else CJK_FONT
        return cand if os.path.exists(cand) else CJK_FONT
    if bold:
        return LATIN_FONT_BOLD if os.path.exists(LATIN_FONT_BOLD) else LATIN_FONT
    return LATIN_FONT if os.path.exists(LATIN_FONT) else LATIN_FONT_BOLD


def _text_stroke_color(img: Image.Image, box) -> tuple[int, int, int]:
    """Average colour of the darker/lighter stroke pixels = original text colour."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    crop = np.array(img.convert("RGB").crop((x0, y0, x1, y1)))
    if crop.size == 0:
        return (0, 0, 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Text is the minority class (fewer pixels than the background).
    fg = th if (th > 0).sum() <= (th == 0).sum() else cv2.bitwise_not(th)
    mask = fg > 0
    if not mask.any():
        return (0, 0, 0)
    return tuple(int(v) for v in crop[mask].mean(axis=0))


def _estimate_bold(img: Image.Image, box) -> bool:
    """Guess whether the original text is bold (heuristic, weight preservation).

    Two cues, either of which flags bold:
      * stroke half-width (distance transform) relative to the text's *core*
        (x-height) band — normalising by the core rather than the full box
        avoids ascenders/descenders skewing the ratio;
      * ink coverage — bold glyphs fill more of their footprint.
    """
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    crop = np.array(img.convert("RGB").crop((x0, y0, x1, y1)))
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (th if (th > 0).sum() <= (th == 0).sum() else cv2.bitwise_not(th)) > 0
    if not fg.any():
        return False
    dist = cv2.distanceTransform(fg.astype("uint8"), cv2.DIST_L2, 3)
    stroke_w = 2.0 * float(dist[fg].mean())
    rows = fg.sum(axis=1)
    core = int((rows > 0.3 * rows.max()).sum()) or (y1 - y0)
    coverage = float(fg.mean())
    return (stroke_w / max(1, core) > 0.16) or (coverage > 0.24)


def _estimate_bg_color(img: Image.Image, box) -> tuple[int, int, int]:
    """Local background colour = median of the (already text-free) box area.

    Call this on the *cleaned* image so the original strokes don't skew it.
    """
    import numpy as np

    x0, y0, x1, y1 = box
    crop = np.array(img.convert("RGB").crop((x0, y0, x1, y1)))
    if crop.size == 0:
        return (255, 255, 255)
    return tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))


@dataclass
class Style:
    color: tuple[int, int, int]   # original text colour
    bold: bool                    # original weight


def region_styles(img: Image.Image, regions: list[Region]) -> list[Style]:
    """Sample each region's original text colour + weight (before removal)."""
    return [Style(color=_text_stroke_color(img, r.box), bold=_estimate_bold(img, r.box))
            for r in regions]


def _region_text_mask(crop, pad: int, box_h: int):
    """Pixel mask of text strokes inside a padded crop, polarity-independent.

    Rather than guessing which Otsu class is text (which fails for coloured text
    or when text fills most of the box), we estimate the *background* colour from
    the crop's border ring and mark any pixel that differs from it — so it works
    regardless of text colour, contrast, or how much of the box the text fills.
    """
    import cv2
    import numpy as np

    ch, cw = crop.shape[:2]
    if ch < 3 or cw < 3:
        return None
    c = crop.astype(np.int16)
    p = max(1, min(pad, ch // 2, cw // 2))
    ring = np.concatenate([
        c[:p].reshape(-1, 3), c[-p:].reshape(-1, 3),
        c[:, :p].reshape(-1, 3), c[:, -p:].reshape(-1, 3),
    ])
    bg = np.median(ring, axis=0)
    dist = np.abs(c - bg).max(axis=2).astype(np.uint8)  # 0..255 distance from bg
    # Otsu on the distance map splits "near bg" from "text".
    _, fg = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close glyph holes and grow strokes in proportion to text size so the
    # inpaint fully covers anti-aliased edges / thin serifs.
    k = int(np.clip(box_h // 10, 2, 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    # Two dilation passes: one pass leaves the faint anti-aliased rim around
    # glyphs unmasked (a ghost outline after inpainting); two reaches it.
    fg = cv2.dilate(fg, kernel, iterations=2)
    return fg


def _uniform_bg_color(crop, pad: int, box_h: int):
    """If the region sits on a solid-colour banner, return that colour (RGB);
    else None (text on a photo / textured background).

    雜誌標題常是「紅字＋白色色塊底」。若沿用整張圖的 inpaint，會把白色色塊
    一起抹掉、換成周圍深色影像，導致 paste 階段誤判背景太暗而把紅字改成白字。
    偵測到色塊底時，remove_text 改為直接用色塊顏色填滿整框（重建色塊），
    paste 階段便會看到淺色背景、保留原本的字色。判定方式：文字以外的背景
    像素若顏色夠一致（標準差低），即視為純色色塊。
    """
    import numpy as np

    fg = _region_text_mask(crop, pad, box_h)
    if fg is None:
        return None
    bg_mask = fg == 0
    if bg_mask.sum() < 0.15 * fg.size:   # 幾乎全是文字 → 無從判斷背景
        return None
    bg_pix = crop[bg_mask]
    if float(bg_pix.std(axis=0).mean()) > 22:   # 背景不一致 → 照片/紋理底
        return None
    return tuple(int(v) for v in np.median(bg_pix, axis=0))


def _gradient_fill(rgb, box):
    """Reconstruct a smooth (vertically graded) background across the box by
    interpolating between the clean background strips just above and below it.
    Returns an RGB patch (box_h × box_w × 3), or None if the surroundings are
    not clean background (a photo / another text line sits in the strips).

    大面積文字（整段介紹文、大標）疊在平滑漸層底（例如灰階雜誌背景）上時，
    TELEA 會把邊緣顏色往內暈成塊狀接縫，純色填滿又會在漸層上留一塊平板。
    改用「取框上下方各數列乾淨背景，逐列垂直內插」重建漸層，銜接自然。
    """
    import numpy as np

    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    s = int(np.clip(bh // 6, 3, 24))            # 參考條厚度
    top = rgb[max(0, y0 - s):y0, x0:x1].astype(np.float32)
    bot = rgb[y1:min(h, y1 + s), x0:x1].astype(np.float32)
    if top.size == 0 or bot.size == 0:
        return None
    # 上下參考條必須是乾淨背景（無文字/物件邊緣）：整條顏色變異要夠低。
    if top.reshape(-1, 3).std(0).mean() > 18 or bot.reshape(-1, 3).std(0).mean() > 18:
        return None
    top_ref = np.median(top, axis=0)            # (bw, 3) 逐列代表色
    bot_ref = np.median(bot, axis=0)
    # 兩條參考色代表的是「條中心」的值（約在框外 s/2 處），而非框緣。內插
    # 係數要對應到真實位置，否則填色的漸層範圍會比框稍寬、露出矩形接縫。
    th = top.shape[0]                            # 實際可用的上條列數
    bh_bot = bot.shape[0]
    span = bh + th / 2 + bh_bot / 2
    a0 = (th / 2) / span
    a1 = (th / 2 + bh) / span
    a = np.linspace(a0, a1, bh, dtype=np.float32)[:, None, None]
    patch = (1 - a) * top_ref[None] + a * bot_ref[None]
    return patch.astype(np.uint8)


def remove_text(img: Image.Image, regions: list[Region]) -> Image.Image:
    """Erase the original text (去除文字) so the translation can be drawn in.

    Three cases per region:
      * 純色色塊底（雜誌標題）→ 直接用色塊顏色填滿整框，重建色塊，讓
        paste 階段看得到淺色背景、保留原字色（見 `_uniform_bg_color`）。
      * 照片/紋理底 → 只遮住筆畫本身送 inpaint，避免整框矩形修復把鄰近
        像素暈染成殘影。
      * 公式或品牌 logo（`_untouched`）→ 完全不動。
    """
    import cv2
    import numpy as np

    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    banners: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]] = []
    grad_fills: list[tuple[tuple[int, int, int, int], "np.ndarray"]] = []

    for r in regions:
        if _untouched(r):
            continue
        x0, y0, x1, y1 = r.box
        box_h = y1 - y0

        # 先在「緊框」上判斷是否為純色色塊底：緊框整個落在色塊內，不會像
        # 加了 padding 的裁切那樣把色塊外的深色背景一起納入而誤判為非純色。
        tx0, ty0 = max(0, x0), max(0, y0)
        tx1, ty1 = min(w, x1), min(h, y1)
        ring_pad = int(np.clip(box_h // 8, 2, 8))
        banner_rgb = _uniform_bg_color(rgb[ty0:ty1, tx0:tx1], ring_pad, box_h)
        if banner_rgb is not None:
            # 純色色塊：整框（略為外擴以蓋住抗鋸齒毛邊）填滿色塊色，
            # 記下來稍後直接貼上（不進 inpaint 遮罩）。
            fp = 4
            banners.append(((max(0, x0 - fp), max(0, y0 - fp),
                             min(w, x1 + fp), min(h, y1 + fp)), banner_rgb))
            continue

        # 平滑漸層底（大面積文字最需要）：用上下乾淨背景逐列內插重建漸層，
        # 避免 TELEA 在大面積上暈成塊狀接縫。
        fp = 3
        gx0, gy0 = max(0, x0 - fp), max(0, y0 - fp)
        gx1, gy1 = min(w, x1 + fp), min(h, y1 + fp)
        patch = _gradient_fill(rgb, (gx0, gy0, gx1, gy1))
        if patch is not None:
            grad_fills.append(((gx0, gy0, gx1, gy1), patch))
            continue

        # 照片/紋理底：外圍留一圈 padding，一方面讓 `_region_text_mask` 有
        # 乾淨的邊界環估計背景色，另一方面涵蓋 OCR 框略小時漏掉的筆畫毛邊。
        pad = int(np.clip(box_h // 4, 4, 16))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
        fg = _region_text_mask(rgb[cy0:cy1, cx0:cx1], pad, box_h)
        if fg is None:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)  # 太小 → 整框遮罩
        else:
            mask[cy0:cy1, cx0:cx1] = np.maximum(mask[cy0:cy1, cx0:cx1], fg)

    # 公式 / 品牌 logo 必須原封不動：鄰近段落的遮罩（含膨脹）若疊到其上，
    # inpaint 會毀掉一部分並留下模糊污漬，這裡強制清除該範圍的遮罩。
    protect_m = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        if _untouched(r):
            x0, y0, x1, y1 = r.box
            protect_m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    mask[protect_m > 0] = 0

    # TELEA 修復會從遮罩「邊界」取樣：緊貼遮罩的保留像素（行內的 1/√dk、dk
    # 等）會被暈進修復區，留下一條條拖影。修復前先把保留區域暫時蓋成周圍
    # 背景色，全部修復完成後再把原始像素貼回去。
    work = bgr
    if protect_m.any():
        work = bgr.copy()
        for r in regions:
            if not _untouched(r):
                continue
            x0, y0, x1, y1 = r.box
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, max(0, x1)), min(h, max(0, y1))
            p = 4
            strips = [bgr[max(0, y0 - p):y0, x0:x1], bgr[y1:y1 + p, x0:x1],
                      bgr[y0:y1, max(0, x0 - p):x0], bgr[y0:y1, x1:x1 + p]]
            ring = [s.reshape(-1, 3) for s in strips if s.size]
            bg = (np.median(np.concatenate(ring), axis=0)
                  if ring else np.array([255, 255, 255]))
            work[y0:y1, x0:x1] = bg.astype(np.uint8)

    radius = max(3, min(h, w) // 150)
    cleaned_bgr = (cv2.inpaint(work, mask, radius, cv2.INPAINT_TELEA)
                   if mask.any() else work.copy())

    def _apply_fills():
        # 重建純色色塊：直接用色塊顏色填滿整框（覆蓋原文字）。
        for (bx0, by0, bx1, by1), color_rgb in banners:
            cleaned_bgr[by0:by1, bx0:bx1] = color_rgb[::-1]  # RGB → BGR
        # 重建平滑漸層：貼上逐列內插的背景（RGB → BGR）。
        for (bx0, by0, bx1, by1), patch in grad_fills:
            cleaned_bgr[by0:by1, bx0:bx1] = patch[:, :, ::-1]

    _apply_fills()

    # 已由色塊/漸層重建的像素，不需再做殘影檢查。
    filled_m = np.zeros((h, w), dtype=np.uint8)
    for (bx0, by0, bx1, by1), _ in banners + grad_fills:
        filled_m[by0:by1, bx0:bx1] = 255

    # 清除品質檢查（無條件執行）：筆畫遮罩偶爾會低估粗筆畫（粗體標題最
    # 常見），inpaint 後留下讀得出來的鬼影；極端情況下遮罩甚至可能全空。
    # 逐區檢查修復結果，仍殘留高對比筆畫的區域退回「整框遮罩」再修一次
    # （保留 / 已重建區域照樣排除）。
    retry = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        if _untouched(r):
            continue
        x0, y0 = max(0, r.box[0]), max(0, r.box[1])
        x1, y1 = max(0, r.box[2]), max(0, r.box[3])
        crop = cleaned_bgr[y0:y1, x0:x1]
        keep = (protect_m[y0:y1, x0:x1] == 0) & (filled_m[y0:y1, x0:x1] == 0)
        if crop.size == 0 or not keep.any():
            continue
        c = crop.astype(np.int16)
        local_bg = np.median(c.reshape(-1, 3), axis=0)
        residual = float(((np.abs(c - local_bg).max(axis=2) > 60) & keep).mean())
        if residual > 0.02:
            cv2.rectangle(retry, (x0, y0), (x1, y1), 255, -1)

    if retry.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        retry = cv2.dilate(retry, kernel)
        retry[protect_m > 0] = 0
        retry[filled_m > 0] = 0
        cleaned_bgr = cv2.inpaint(cleaned_bgr, retry, radius, cv2.INPAINT_TELEA)
        _apply_fills()   # 重修後再蓋回色塊 / 漸層

    # 把保留區域的原始像素貼回（前面為了避免拖影暫時用背景色蓋住了）。
    if protect_m.any():
        cleaned_bgr[protect_m > 0] = bgr[protect_m > 0]

    rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# CJK 表意文字、假名、全形標點等「任意處皆可斷行」的字元。
_CJK_CHAR = (
    r"[⺀-〿぀-ヿ㐀-䶿一-鿿"
    r"豈-﫿＀-￯]"
)
# 斷行單位：一個 CJK 字元自成一個單位；連續的非 CJK 非空白字元（英文單字、
# 數字、符號）整串為一個單位。捕捉前面的空白以決定要不要補空格。
_WRAP_TOKEN_RE = re.compile(rf"(\s*)({_CJK_CHAR}|(?:(?!{_CJK_CHAR})\S)+)")


def _wrap(draw, text, font, max_w):
    """Greedy wrap so each line fits max_w. Returns (lines, h, w).

    Latin words stay whole and rejoin with a space; CJK characters may break
    anywhere. Wrapping must be token-based, not "split on spaces if any":
    a Chinese translation with an embedded Latin word (e.g. "…中間有一個
    ReLU 啟用…") contains spaces, and space-splitting would treat each long
    Chinese run as one unbreakable word — the font would then have to shrink
    until that entire run fits on a single line.
    """
    lines: list[str] = []
    cur = ""
    for sep, unit in _WRAP_TOKEN_RE.findall(text):
        # 超過整行寬度的單一單位（超長英文字串等）退回逐字元硬切。
        pieces = ([unit] if len(unit) <= 1
                  or draw.textlength(unit, font=font) <= max_w else list(unit))
        for i, piece in enumerate(pieces):
            joint = " " if (sep and cur and i == 0) else ""
            trial = cur + joint + piece
            if not cur or draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = piece
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    total_h = line_h * len(lines)
    max_line_w = max((draw.textlength(ln, font=font) for ln in lines), default=0)
    return lines, total_h, max_line_w


def _fit_font(draw, text, font_path, box_w, box_h, max_size):
    """Largest font size whose wrapped text fits the box. Returns (font,lines,h)."""
    lo, hi, best = 6, max(6, max_size), None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        lines, total_h, max_w = _wrap(draw, text, font, box_w)
        if max_w <= box_w and total_h <= box_h:
            best = (font, lines, total_h)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = ImageFont.truetype(font_path, 6)
        lines, total_h, _ = _wrap(draw, text, font, box_w)
        best = (font, lines, total_h)
    return best


def _luminance(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _color_dist(a, b) -> float:
    """Euclidean distance in RGB. Used for the legibility check instead of a
    pure luminance difference: a red title on a dark-green background has a
    small *luminance* gap but a large *colour* gap, and it is perfectly
    legible — so we should keep the original colour rather than flipping it
    to white. Luminance alone wrongly flagged such designs as illegible.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def draw_ocr_overlay(img: Image.Image, regions: list[Region]) -> Image.Image:
    """Step 1 visual — draw the detected boxes + recognised text on the image.

    Red = normal text (will be erased + translated). Green = formula, left
    untouched. Blue = brand masthead/logo, kept untranslated.
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    label_font = ImageFont.truetype(
        LATIN_FONT if os.path.exists(LATIN_FONT) else _pick_font_path("en"), 12)
    for r in regions:
        x0, y0 = r.box[0], r.box[1]
        color = ((37, 99, 235) if r.keep
                 else (16, 163, 74) if r.is_formula else (255, 0, 0))
        draw.rectangle(r.box, outline=color, width=2)
        # Put the recognised text just above the box on a small chip.
        text = r.original
        tw = draw.textlength(text, font=label_font)
        ly = max(0, y0 - 14)
        draw.rectangle((x0, ly, x0 + tw + 4, ly + 14), fill=color)
        draw.text((x0 + 2, ly + 1), text, font=label_font, fill=(255, 255, 255))
    return img


def paste_translations(
    img: Image.Image,
    regions: list[Region],
    styles: list[Style],
    target_language: str,
    show_boxes: bool = False,
) -> Image.Image:
    """Steps 3/4 visual — draw the translated text into each box.

    Preserves the original weight (bold), reuses the original text colour, and
    blends into the local background: the outline is drawn in the estimated
    background colour so each glyph sits on a locally-uniform patch (natural on
    busy backgrounds). If the original colour barely contrasts with the new
    background, we fall back to black/white so it stays legible.

    `show_boxes=True` outlines each box (translation-mapping view); False gives
    the final clean paste.
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    for r, style in zip(regions, styles):
        if _untouched(r):
            # Formula / brand-logo pixels were never erased either — leave
            # them exactly as in the original image, don't draw over them.
            continue

        x0, y0, x1, y1 = r.box
        box_w, box_h = x1 - x0, y1 - y0

        if show_boxes:
            draw.rectangle(r.box, outline=(37, 99, 235), width=1)

        # Auto-estimate the local background colour (on the cleaned image) and
        # ensure the text stays readable against it. Use colour distance, not
        # a luminance gap: a red title on dark green is legible (large colour
        # gap) even though the luminance gap is small — keep the original
        # colour. Only fall back to black/white when the colour is genuinely
        # too close to the background to read.
        bg = _estimate_bg_color(img, r.box)
        text_color = style.color
        if _color_dist(text_color, bg) < 75:
            text_color = (0, 0, 0) if _luminance(bg) > 128 else (255, 255, 255)

        # Preserve the original weight when choosing the font.
        font_path = _pick_font_path(target_language, bold=style.bold)
        font, lines, total_h = _fit_font(
            draw, r.translation, font_path, box_w, box_h, max_size=box_h)
        # Thin background-coloured halo → the text fuses with the background.
        stroke_w = max(1, font.size // 14)

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        y = y0 + max(0, (box_h - total_h) // 2)
        for ln in lines:
            lw = draw.textlength(ln, font=font)
            x = x0 + max(0, (box_w - lw) // 2)
            draw.text((x, y), ln, font=font, fill=text_color,
                      stroke_width=stroke_w, stroke_fill=bg)
            y += line_h

    return img


def render(img: Image.Image, regions: list[Region], target_language: str) -> Image.Image:
    """Full pipeline → final image (去除文字 + 翻譯貼上)."""
    img = img.convert("RGB")
    # Sample the original text colour + weight *before* erasing it.
    styles = region_styles(img, regions)
    cleaned = remove_text(img, regions)
    return paste_translations(cleaned, regions, styles, target_language)


def pipeline_steps(
    img: Image.Image, target_language: str, source_langs: tuple[str, ...]
) -> tuple[list[Region], list[tuple[str, Image.Image]]]:
    """Run the whole pipeline and return every intermediate stage as an image."""
    img = img.convert("RGB")
    regions = detect_and_translate(img, target_language, source_langs)
    styles = region_styles(img, regions)

    step1 = draw_ocr_overlay(img, regions)            # OCR 抓到文字
    step2 = remove_text(img, regions)                 # 去除文字
    step3 = paste_translations(step2.copy(), regions, styles,
                               target_language, show_boxes=True)  # 文字翻譯
    step4 = paste_translations(step2.copy(), regions, styles,
                               target_language, show_boxes=False)  # 文字貼上

    steps = [
        ("1 · Detect text (OCR)", step1),
        ("2 · Remove text (inpaint)", step2),
        ("3 · Translate (map to boxes)", step3),
        ("4 · Paste translation (final)", step4),
    ]
    return regions, steps


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "engine": "paddleocr", "gpu": USE_GPU}


@app.post("/translate")
async def translate(
    image: UploadFile = File(...),
    target_language: str = Form(...),
    source_languages: str = Form(""),        # comma list, e.g. "en,fr"; empty = auto-detect
    response_format: str = Form("image"),    # "image" or "json"
):
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty image upload.")
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read image: {exc}") from exc

    # NOTE: no `or ("en",)` fallback here — an empty tuple means "auto-detect"
    # and must stay distinguishable from an explicit "en" choice (they now
    # drive different things: OCR model selection AND the translator's
    # source language). `_paddle_lang(())` already defaults to "en" for OCR.
    src = tuple(s.strip() for s in source_languages.split(",") if s.strip())

    try:
        regions = detect_and_translate(pil, target_language, src)
    except ValueError as exc:  # e.g. unsupported EasyOCR language combo
        raise HTTPException(400, str(exc)) from exc

    out = render(pil, regions, target_language)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    png = buf.getvalue()

    if response_format == "json":
        import base64
        return JSONResponse({
            "target_language": target_language,
            "regions": [
                {"box": r.box, "original": r.original, "translation": r.translation}
                for r in regions
            ],
            "image_base64": base64.b64encode(png).decode("ascii"),
        })

    return Response(content=png, media_type="image/png")


@app.post("/steps")
async def steps(
    image: UploadFile = File(...),
    target_language: str = Form(...),
    source_languages: str = Form(""),   # empty = auto-detect for both OCR model
                                         # (falls back to the broadest "en" model)
                                         # and Google Translate's source language
):
    """Return every intermediate stage of the pipeline as base64 PNGs."""
    import base64

    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty image upload.")
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read image: {exc}") from exc

    src = tuple(s.strip() for s in source_languages.split(",") if s.strip())

    try:
        regions, stages = pipeline_steps(pil, target_language, src)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    def b64(im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse({
        "target_language": target_language,
        "regions": [
            {"box": r.box, "original": r.original, "translation": r.translation}
            for r in regions
        ],
        "steps": [{"name": name, "image": b64(im)} for name, im in stages],
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Translation (local OCR)</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
         padding: 0 1rem; color: #222; }
  h1 { font-size: 1.4rem; }
  form { display: grid; gap: .75rem; padding: 1rem; border: 1px solid #ddd;
         border-radius: 10px; }
  label { font-weight: 600; font-size: .9rem; }
  input, select, button { padding: .5rem; font-size: 1rem; }
  select { width: 100%; }
  button { background: #16a34a; color: #fff; border: 0; border-radius: 8px;
           cursor: pointer; }
  button:disabled { opacity: .6; cursor: progress; }
  #out { margin-top: 1.25rem; }
  .status { color: #666; font-size: .9rem; }
  .steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
           gap: 1rem; margin-top: 1rem; }
  .step { border: 1px solid #eee; border-radius: 10px; padding: .5rem;
          background: #fafafa; }
  .step h3 { margin: .1rem 0 .5rem; font-size: .95rem; }
  .step img { max-width: 100%; border-radius: 6px; display: block; }
  table.regions { border-collapse: collapse; margin-top: 1rem; font-size: .85rem;
                  width: 100%; }
  table.regions th, table.regions td { border: 1px solid #eee; padding: .3rem .5rem;
                                       text-align: left; }
</style>
</head>
<body>
<h1>Image Translation <small>(local OCR, no API key)</small></h1>
<p class="status">PaddleOCR detects text → remove text (inpaint) → translate → paste
translation. Each stage is shown below.</p>
<form id="f">
  <div>
    <label>Image</label><br>
    <input type="file" name="image" accept="image/*" required>
  </div>
  <div>
    <label>Source language (text in the image)</label><br>
    <select name="source_languages">
      <option value="" selected>Auto-detect (recommended)</option>
      <option value="en">English</option>
      <option value="fr">French</option>
      <option value="de">German</option>
      <option value="es">Spanish</option>
      <option value="zh">Simplified Chinese</option>
      <option value="zh-TW">Traditional Chinese</option>
      <option value="ja">Japanese</option>
      <option value="ko">Korean</option>
    </select>
    <p class="status" style="margin:.3rem 0 0;">Auto-detect works well for full
      sentences. If a page has lots of short/standalone words or legal terms
      (where Google's language auto-detect can guess wrong), pick the actual
      language here for more reliable translation.</p>
  </div>
  <div>
    <label>Target language</label><br>
    <select name="target_language" required>
      <option value="Traditional Chinese" selected>Traditional Chinese</option>
      <option value="Simplified Chinese">Simplified Chinese</option>
      <option value="English">English</option>
      <option value="Japanese">Japanese</option>
      <option value="Korean">Korean</option>
      <option value="French">French</option>
      <option value="German">German</option>
      <option value="Spanish">Spanish</option>
    </select>
  </div>
  <button type="submit">Translate (show steps)</button>
</form>
<div id="out"></div>
<script>
const f = document.getElementById('f');
const out = document.getElementById('out');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = f.querySelector('button');
  btn.disabled = true; btn.textContent = 'Translating…';
  out.innerHTML = '<p class="status">Working… (first run downloads OCR models)</p>';
  try {
    const res = await fetch('/steps', { method: 'POST', body: new FormData(f) });
    if (!res.ok) {
      out.innerHTML = '<p class="status">Error: ' + (await res.text()) + '</p>';
      return;
    }
    const data = await res.json();
    const grid = data.steps.map(s =>
      `<div class="step"><h3>${s.name}</h3><img src="${s.image}"></div>`).join('');
    const rows = data.regions.map(r =>
      `<tr><td>${escapeHtml(r.original)}</td><td>${escapeHtml(r.translation)}</td></tr>`).join('');
    out.innerHTML =
      `<div class="steps">${grid}</div>` +
      `<table class="regions"><thead><tr><th>original</th><th>translation</th></tr></thead>` +
      `<tbody>${rows}</tbody></table>`;
  } catch (err) {
    out.innerHTML = '<p class="status">Error: ' + err + '</p>';
  } finally {
    btn.disabled = false; btn.textContent = 'Translate (show steps)';
  }
});
function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
</script>
</body>
</html>"""