"""Issue 16: cv2.imwrite fails on Cyrillic paths; imwrite_unicode must not."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ir import BBox
from ingestion.preprocess import imwrite_unicode
from ingestion.region_detect import is_valid_figure_bbox
from ingestion.table_pipeline import _FIGURE_CAPTION_RE, find_nearby_figure_caption


def test_crop_save_cyrillic() -> int:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (20, 80, 160)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for doc_id in ["_1954", "РДК_молниезащита 89г", "КТ-160G-14G"]:
            p = root / "figures" / doc_id / "test.png"
            ok = imwrite_unicode(p, img)
            exists = p.is_file() and p.stat().st_size > 0
            mark = "PASS" if ok and exists else "FAIL"
            failed += mark == "FAIL"
            print(f"  {mark}  save {doc_id!r}")
            if exists:
                raw = np.fromfile(str(p), dtype=np.uint8)
                decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape[0] < 10:
                    print(f"  FAIL  decode {doc_id!r}")
                    failed += 1
    return failed


def test_figure_bbox() -> int:
    cases = [
        ("full rdk p1", BBox(x1=0, y1=0, x2=371, y2=791), 378.0, 807.0, False),
        ("full page", BBox(x1=0, y1=10, x2=605, y2=804), 605.0, 814.0, False),
        ("tiny", BBox(x1=10, y1=10, x2=40, y2=40), 595.0, 842.0, False),
        ("ribbon", BBox(x1=10, y1=10, x2=500, y2=30), 595.0, 842.0, False),
        ("normal chart", BBox(x1=80, y1=120, x2=420, y2=420), 595.0, 842.0, True),
    ]
    failed = 0
    for name, bb, page_w, page_h, expect_ok in cases:
        ok, reason = is_valid_figure_bbox(bb, page_w=page_w, page_h=page_h)
        # full-page cases may hit too_large before full_page
        if expect_ok:
            good = ok and reason == ""
        else:
            good = (not ok) and reason in {"too_large", "full_page", "too_small", "bad_aspect"}
        mark = "PASS" if good else "FAIL"
        failed += not good
        print(f"  {mark}  bbox {name:16s} ok={ok} reason={reason}")
    return failed


class _Span:
    def __init__(self, text: str, bbox: BBox) -> None:
        self.text = text
        self.bbox = bbox


def test_caption_spans() -> int:
    fig = BBox(x1=80, y1=100, x2=400, y2=360)
    spans = [
        _Span("обычный текст", BBox(x1=80, y1=370, x2=300, y2=390)),
        _Span("Фиг. I.1. Одновинтовой вертолет", BBox(x1=90, y1=365, x2=380, y2=385)),
    ]
    got = find_nearby_figure_caption(spans, fig)
    far = find_nearby_figure_caption(
        [_Span("Фиг. 9", BBox(x1=90, y1=700, x2=200, y2=720))], fig
    )
    failed = 0
    if "Фиг" not in got:
        print(f"  FAIL  nearby caption got={got!r}")
        failed += 1
    else:
        print(f"  PASS  nearby caption {got[:40]!r}")
    if far:
        print(f"  FAIL  distant caption leaked {far!r}")
        failed += 1
    else:
        print("  PASS  distant caption ignored")
    if not _FIGURE_CAPTION_RE.search("Fig. 2.3 axis"):
        print("  FAIL  Fig regex")
        failed += 1
    else:
        print("  PASS  Fig regex")
    return failed


def main() -> int:
    failed = test_crop_save_cyrillic() + test_figure_bbox() + test_caption_spans()
    print(f"issue16-18 checks: {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
