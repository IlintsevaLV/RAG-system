"""Image quality (IQS) and preprocess routes GOOD / MEDIUM / BAD.

Binarization is off by default (roadmap_new / measured OCR regressions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import fitz
import numpy as np

from ingestion.models import ImageQualityLevel, IQSComponents, IQSResult, PageRoute


class PreprocessStep(str, Enum):
    RENDER = "render"
    DESKEW = "deskew"
    MILD_DENOISE = "mild_denoise"
    DENOISE = "denoise"
    CONTRAST_NORM = "contrast_normalization"
    CONTRAST_ENHANCE = "contrast_enhancement"
    HIGH_RES_BANDS = "high_resolution_bands"
    BINARIZE = "binarize"  # experimental only


@dataclass
class RenderedPage:
    image_bgr: np.ndarray
    dpi: int
    page: int
    width_pt: float
    height_pt: float


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def render_page(
    doc: fitz.Document,
    page_number: int,
    *,
    dpi: int = 200,
) -> RenderedPage:
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif pix.n == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return RenderedPage(
        image_bgr=bgr,
        dpi=dpi,
        page=page_number,
        width_pt=float(page.rect.width),
        height_pt=float(page.rect.height),
    )


def estimate_skew_deg(gray: np.ndarray) -> float:
    """Estimate skew via minAreaRect on ink edges; degrees, positive = CCW."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.shape[0] < 200:
        return 0.0
    # OpenCV where → (row, col) = (y, x)
    pts = coords[:, ::-1].astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45:
        angle = 90 + angle
    # Keep small deskew angles only
    if abs(angle) > 15:
        return 0.0
    return float(angle)


def compute_iqs(image_bgr: np.ndarray, *, dpi: int) -> IQSResult:
    notes: list[str] = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Resolution: prefer >= 200 DPI effective for OCR
    res = _clamp01(dpi / 250.0) if dpi else _clamp01(min(w, h) / 1500.0)
    if dpi < 150:
        notes.append("low render dpi")

    # Blur: Laplacian variance (higher = sharper)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = _clamp01(lap_var / 200.0)
    if lap_var < 40:
        notes.append(f"blurry lap_var={lap_var:.1f}")

    # Noise: high-frequency residual energy
    den = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - den.astype(np.float32)
    noise_std = float(residual.std())
    # moderate noise OK; very high is bad
    noise = _clamp01(1.0 - (noise_std - 5.0) / 25.0)

    # Contrast: usable dynamic range
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = _clamp01(float(p95 - p5) / 180.0)
    if p95 - p5 < 40:
        notes.append("low contrast")

    skew_deg = estimate_skew_deg(gray)
    skew = _clamp01(1.0 - abs(skew_deg) / 8.0)

    # Compression artifacts proxy: blockiness via 8x8 grid energy (JPEG-like)
    # Downscale then measure high-freq in 8x8 DCT-ish difference
    small = cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    g = small.astype(np.float32)
    block = 8
    if g.shape[0] >= block and g.shape[1] >= block:
        diffs = []
        for y in range(0, g.shape[0] - block, block):
            for x in range(0, g.shape[1] - block, block):
                tile = g[y : y + block, x : x + block]
                # mean abs diff across block borders
                diffs.append(float(np.mean(np.abs(tile[:, 0] - tile[:, -1]))))
                diffs.append(float(np.mean(np.abs(tile[0, :] - tile[-1, :]))))
        blockiness = float(np.mean(diffs)) if diffs else 0.0
        compression = _clamp01(1.0 - blockiness / 40.0)
    else:
        compression = 0.7

    components = IQSComponents(
        resolution_score=res,
        blur_score=blur,
        noise_score=noise,
        contrast_score=contrast,
        skew_score=skew,
        compression_score=compression,
    )
    score = _clamp01(
        0.20 * res
        + 0.25 * blur
        + 0.15 * noise
        + 0.20 * contrast
        + 0.10 * skew
        + 0.10 * compression
    )

    if score >= 0.72:
        level = ImageQualityLevel.GOOD
    elif score >= 0.45:
        level = ImageQualityLevel.MEDIUM
    else:
        level = ImageQualityLevel.BAD

    return IQSResult(
        score=score,
        level=level,
        components=components,
        estimated_skew_deg=skew_deg,
        notes=notes,
    )


def preprocess_plan(level: ImageQualityLevel, *, enable_binarize: bool = False) -> list[str]:
    if level == ImageQualityLevel.GOOD:
        steps = [PreprocessStep.RENDER.value, PreprocessStep.DESKEW.value]
    elif level == ImageQualityLevel.MEDIUM:
        steps = [
            PreprocessStep.RENDER.value,
            PreprocessStep.DESKEW.value,
            PreprocessStep.MILD_DENOISE.value,
            PreprocessStep.CONTRAST_NORM.value,
        ]
    else:
        steps = [
            PreprocessStep.RENDER.value,
            PreprocessStep.DESKEW.value,
            PreprocessStep.DENOISE.value,
            PreprocessStep.CONTRAST_ENHANCE.value,
            PreprocessStep.HIGH_RES_BANDS.value,
        ]
    if enable_binarize:
        steps.append(PreprocessStep.BINARIZE.value)
    return steps


def deskew(image_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.15:
        return image_bgr
    h, w = image_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def mild_denoise(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, 3, 3, 7, 21)


def strong_denoise(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, 6, 6, 7, 21)


def contrast_normalize(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def contrast_enhance(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def binarize_experimental(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10
    )
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def horizontal_bands(
    image_bgr: np.ndarray,
    *,
    band_height: int = 1600,
    overlap: float = 0.12,
) -> list[tuple[int, int, np.ndarray]]:
    """Full-width horizontal bands (preferred over square tiles)."""
    h, w = image_bgr.shape[:2]
    if h <= band_height:
        return [(0, h, image_bgr)]
    step = max(1, int(band_height * (1.0 - overlap)))
    bands: list[tuple[int, int, np.ndarray]] = []
    y = 0
    while y < h:
        y2 = min(h, y + band_height)
        bands.append((y, y2, image_bgr[y:y2, :, :].copy()))
        if y2 >= h:
            break
        y += step
    return bands


def apply_preprocess(
    image_bgr: np.ndarray,
    iqs: IQSResult,
    *,
    enable_binarize: bool = False,
) -> tuple[np.ndarray, list[str], list[tuple[int, int, np.ndarray]] | None]:
    """Apply route for IQS level. Returns (image, steps, bands|None)."""
    steps = preprocess_plan(iqs.level, enable_binarize=enable_binarize)
    out = image_bgr
    applied: list[str] = []
    bands = None

    for step in steps:
        if step == PreprocessStep.RENDER.value:
            applied.append(step)
            continue
        if step == PreprocessStep.DESKEW.value:
            out = deskew(out, iqs.estimated_skew_deg)
            applied.append(step)
        elif step == PreprocessStep.MILD_DENOISE.value:
            out = mild_denoise(out)
            applied.append(step)
        elif step == PreprocessStep.DENOISE.value:
            out = strong_denoise(out)
            applied.append(step)
        elif step == PreprocessStep.CONTRAST_NORM.value:
            out = contrast_normalize(out)
            applied.append(step)
        elif step == PreprocessStep.CONTRAST_ENHANCE.value:
            out = contrast_enhance(out)
            applied.append(step)
        elif step == PreprocessStep.HIGH_RES_BANDS.value:
            bands = horizontal_bands(out)
            applied.append(step)
        elif step == PreprocessStep.BINARIZE.value:
            out = binarize_experimental(out)
            applied.append(step)

    return out, applied, bands


def process_page_image(
    doc: fitz.Document,
    page_number: int,
    *,
    route: PageRoute,
    dpi_good: int = 200,
    dpi_bad: int = 300,
    enable_binarize: bool = False,
) -> dict:
    """Render (if OCR route), score IQS, preprocess. Text-layer route skips heavy CV."""
    if route == PageRoute.TEXT_LAYER:
        return {
            "page": page_number,
            "route": route.value,
            "skipped_preprocess": True,
            "reason": "TLQ accepted text layer",
            "iqs": None,
            "steps": [],
        }

    # First pass at normal DPI to score; BAD may re-render higher
    rendered = render_page(doc, page_number, dpi=dpi_good)
    iqs = compute_iqs(rendered.image_bgr, dpi=rendered.dpi)
    if iqs.level == ImageQualityLevel.BAD:
        rendered = render_page(doc, page_number, dpi=dpi_bad)
        iqs = compute_iqs(rendered.image_bgr, dpi=rendered.dpi)

    image, steps, bands = apply_preprocess(
        rendered.image_bgr, iqs, enable_binarize=enable_binarize
    )
    return {
        "page": page_number,
        "route": route.value,
        "skipped_preprocess": False,
        "iqs": iqs,
        "steps": steps,
        "image_shape": list(image.shape),
        "band_count": len(bands) if bands else 0,
        "image_bgr": image,
        "bands": bands,
    }


def save_preview(image_bgr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
