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
    if len(t) < 60 and (1 - letters / len(t)) > 0.35:  # high symbol/digit ratio
        return True
    return False


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
      * the vertical gap to the previous line is small relative to that
        line's own height (ordinary line spacing, not a gap between
        unrelated blocks/paragraphs)
      * the two lines' horizontal ranges overlap substantially (same
        column/block — stops an unrelated sidebar element from merging in)
      * the two lines have similar height (same font size — stops a large
        title merging with body text underneath it)
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
        avg_h = (h + ph) / 2 or 1

        vertical_gap = y0 - py1
        height_ratio = max(h, ph) / max(1, min(h, ph))
        overlap = min(x1, px1) - max(x0, px0)
        overlap_ratio = overlap / max(1, min(x1 - x0, px1 - px0))

        same_paragraph = (
            vertical_gap < 0.7 * avg_h
            and height_ratio < 1.6
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
        if not text or conf < 0.2:
            continue
        pts = np.asarray(poly) / scale
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        raw_lines.append({"text": text, "box": (x0, y0, x1, y1)})

    if not raw_lines:
        return []

    # 2. Group consecutive natural-language lines into paragraphs so they get
    #    translated with full sentence context; formula lines stay isolated.
    groups = _group_into_paragraphs(raw_lines)

    target_code = normalize_target(target_language)
    translator = GoogleTranslator(source=_google_source_lang(source_langs), target=target_code)

    regions: list[Region] = []
    for group in groups:
        is_formula = group[0].get("is_formula", False)
        xs0 = [ln["box"][0] for ln in group]
        ys0 = [ln["box"][1] for ln in group]
        xs1 = [ln["box"][2] for ln in group]
        ys1 = [ln["box"][3] for ln in group]
        box = (min(xs0), min(ys0), max(xs1), max(ys1))
        original = " ".join(ln["text"] for ln in group)

        if is_formula:
            # Leave formulas completely untouched downstream: no inpaint,
            # no redraw (see is_formula handling in remove_text/paste_translations).
            regions.append(Region(box=box, original=original,
                                  translation=original, is_formula=True))
            continue

        try:
            translation = translator.translate(original) or original
        except Exception:  # noqa: BLE001 - fall back to original on failure
            translation = original
        regions.append(Region(box=box, original=original,
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


def remove_text(img: Image.Image, regions: list[Region]) -> Image.Image:
    """Erase the original text (去除文字) by inpainting the full bounding boxes.
    直接使用與 Notebook 相同的邏輯：將 OCR 的文字方塊轉為遮罩並膨脹，再送入 OpenCV 修復。
    """
    import cv2
    import numpy as np

    # 1. 將 Pillow 影像轉換為 OpenCV 的 BGR 格式
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    
    # 2. 建立一個與原圖一樣大小的黑底遮罩 (Mask)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 3. 將每個文字方塊（Bounding Box）直接填滿白色（公式區域跳過，保持原樣）
    for r in regions:
        if r.is_formula:
            continue
        x0, y0, x1, y1 = r.box
        # 在遮罩上繪製填滿的矩形方塊
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    # 4. 如果有偵測到任何文字，進行遮罩膨脹與影像修復
    if mask.any():
        # 建立一個 7x7 的結構元素進行膨脹，確保文字外圍的毛邊與陰影也被完整覆蓋
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(mask, kernel)
        
        # 執行 OpenCV Inpaint (此處使用 TELEA，你也可以換成腳本原本的 cv2.INPAINT_NS)
        radius = max(3, min(h, w) // 150)
        cleaned_bgr = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
        
        # 將結果轉回 RGB
        rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(rgb)


def _wrap(draw, text, font, max_w):
    """Greedy word/char wrap so each line fits max_w. Returns (lines, h, w)."""
    lines: list[str] = []
    if " " in text:
        cur = ""
        for w in text.split():
            trial = f"{cur} {w}".strip()
            if not cur or draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    else:
        cur = ""
        for ch in text:
            if not cur or draw.textlength(cur + ch, font=font) <= max_w:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
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


def draw_ocr_overlay(img: Image.Image, regions: list[Region]) -> Image.Image:
    """Step 1 visual — draw the detected boxes + recognised text on the image.

    Red = normal text (will be erased + translated). Green = detected as a
    formula/equation and will be left completely untouched.
    """
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    label_font = ImageFont.truetype(
        LATIN_FONT if os.path.exists(LATIN_FONT) else _pick_font_path("en"), 12)
    for r in regions:
        x0, y0 = r.box[0], r.box[1]
        color = (16, 163, 74) if r.is_formula else (255, 0, 0)
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
        if r.is_formula:
            # Formula pixels were never erased either — leave them exactly
            # as in the original image, don't draw anything over them.
            continue

        x0, y0, x1, y1 = r.box
        box_w, box_h = x1 - x0, y1 - y0

        if show_boxes:
            draw.rectangle(r.box, outline=(37, 99, 235), width=1)

        # Auto-estimate the local background colour (on the cleaned image) and
        # ensure the text stays readable against it.
        bg = _estimate_bg_color(img, r.box)
        text_color = style.color
        if abs(_luminance(text_color) - _luminance(bg)) < 55:
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