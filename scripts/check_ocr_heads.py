"""Checks for multi-head OCR merging (Cyrillic + Latin + Greek).

1. Unit cases: overlapping head readings -> expected chosen text.
2. Replay: raw head outputs cached by `--capture` are merged again and every line that
   differs from the Cyrillic head is printed, with counts of garbled Greek lines.

Usage:
  python scripts/check_ocr_heads.py
  python scripts/check_ocr_heads.py --capture "<pdf>" --tag kt160g --pages 1,4,7
  python scripts/check_ocr_heads.py --replay kt160g
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.tech_symbols import choose_ocr_candidate, is_garbled_greek

CACHE = ROOT / "data" / "cache" / "ocr_heads"

C, L, G = "cyrillic", "latin", "el"
# (candidates, expected text)
UNIT_CASES: list[tuple[list[tuple[str, float, str]], str]] = [
    ([("КНИГА И.Общие положения,", 0.95, C), ("ΚΗΑΓΑ Ι. Ο6μΜΕ ποπΟΣΚεΗμα,", 0.74, G)], "КНИГА И.Общие положения,"),
    ([("Этап 2:", 0.99, C), ("Θταπ 2:", 0.86, G)], "Этап 2:"),
    ([("ОБОРУДОВАНИЯ", 1.00, C), ("ΟΒΟΡΥΔΟΒΑΗΚΑ", 0.81, G)], "ОБОРУДОВАНИЯ"),
    ([("температуру", 0.93, C), ("memnepamypy", 0.97, L), ("ΤΕΜΠΕΡΑΤΥΡΥ", 0.80, G)], "температуру"),
    ([("если", 0.76, C), ("ecnu", 0.95, L)], "если"),
    ([("ВЫсоКуЮ", 0.56, C), ("BICOKYO", 0.87, L)], "ВЫсоКуЮ"),
    ([("Категория A -", 0.95, C), ("Kaτeropus Α -", 0.88, G)], "Категория A -"),
    ([("Испытание 1: f = 11 Гц", 0.90, C), ("Ucπβιταημe 1: φ = 11 Γu", 0.82, G)], "Испытание 1: f = 11 Гц"),
    ([("B", 0.99, C), ("Β", 0.77, G)], "B"),
    ([("7", 0.62, C), ("Ο", 0.54, G)], "7"),
    ([("KT 160Г/14Г", 0.80, C), ("KT 160Γ/14Γ", 0.85, G)], "KT 160Г/14Г"),
    ([("The aircraft shall", 0.60, C), ("The aircraft shall", 0.95, L)], "The aircraft shall"),
    # isolated Greek symbols inside Russian text come from the Greek head
    ([("где ш — угловая скорость", 0.90, C), ("γπε ω — γγλοβαη σκοροστь", 0.80, G)], "где ω — угловая скорость"),
    ([("при а = 5°", 0.90, C), ("πρι α = 5°", 0.80, G)], "при α = 5°"),
    ([("а также", 0.90, C), ("α τακΚε", 0.80, G)], "а также"),
    ([("таблицей 4.1 для испытаний", 0.95, C), ("ταβλμμεη 4.1 ΔI ΜCΠΕΙΤΑΗΗΗ", 0.80, G)], "таблицей 4.1 для испытаний"),
    ([("зоны 1а и зоны 2.", 0.95, C), ("30ΗΕΙ 1a ν 30ΗΕΙ 2.", 0.80, G)], "зоны 1а и зоны 2."),
    ([("указанному в п. 9.6.2.", 0.95, C), ("yκα3αΗΗομγ Β π. 9.6.2.", 0.80, G)], "указанному в п. 9.6.2."),
    ([("Крылов А.Н.", 0.86, C), ("K A.H.", 0.90, L)], "Крылов А.Н."),
    ([("Категория, п.4.3", 0.95, C), ("Κατεγορια, π.4.3", 0.85, G)], "Категория, п.4.3"),
    ([("П.4.6.1", 0.94, C), ("N.4.6.1", 1.00, L), ("Π.4.6.1", 1.00, G)], "П.4.6.1"),
    ([("СЛОИСтЫХ", 0.67, C), ("CJOKCTLIX", 0.85, L)], "СЛОИСтЫХ"),
    ([("вызываЮт", 0.68, C), ("BLI3LIBAIOT", 0.92, L)], "вызываЮт"),
    # formula lines keep the Greek head
    ([("dMк=к 1 р(шr)2", 0.70, C), ("dMx=κ 1 ρ(ωr)²", 0.80, G)], "dMx=κ 1 ρ(ωr)²"),
    ([("шR", 0.60, C), ("ωR", 0.80, G)], "ωR"),
    ([("CпR2р(шR)2", 0.60, C), ("CπR2ρ(ωR)2", 0.80, G)], "CπR2ρ(ωR)2"),
    ([("75 N = cN πR2  (ωR) .", 0.70, C), ("75N =CNπR2 ρ (ωR)².", 0.85, G)], "75N =CNπR2 ρ (ωR)²."),
    ([("сопротивление 8. Первая часть", 0.95, C), ("σοπροτηβλεΗμε δ. Πεpβαη ψαστь", 0.80, G)], "сопротивление δ. Первая часть"),
]


def run_unit_cases() -> int:
    failed = 0
    for cands, expected in UNIT_CASES:
        got = choose_ocr_candidate(cands)[0]
        ok = got == expected
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {cands[0][0]!r:32s} -> {got!r}" + ("" if ok else f"  expected {expected!r}"))
    print(f"unit cases: {len(UNIT_CASES) - failed}/{len(UNIT_CASES)} passed")
    return failed


def capture(pdf: str, tag: str, pages: list[int], dpi: int) -> None:
    import fitz

    import ingestion.ocr_rapid as ocr
    from ingestion.preprocess import render_page

    captured: list[dict] = []
    orig = ocr._merge_multi_head_rows

    def _capture(heads):
        captured.append({k: list(v) for k, v in heads.items()})
        return orig(heads)

    ocr._merge_multi_head_rows = _capture
    CACHE.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    try:
        for page in pages:
            path = CACHE / f"{tag}_p{page:03d}.pkl"
            if path.exists():
                continue
            captured.clear()
            img = render_page(doc, page, dpi=dpi).image_bgr
            ocr.recognize_page_image(img, dpi=dpi, page=page, doc_id=tag)
            path.write_bytes(pickle.dumps(captured))
            print(f"captured {path.name}")
    finally:
        ocr._merge_multi_head_rows = orig
        doc.close()


def replay(tag: str) -> None:
    from ingestion.ocr_rapid import _merge_multi_head_rows

    n_lines = n_changed = n_garbled = 0
    for path in sorted(CACHE.glob(f"{tag}_p*.pkl")):
        page = path.stem.rsplit("_p", 1)[1]
        for heads in pickle.loads(path.read_bytes()):
            cyr = heads.get("cyrillic", [])
            merged = _merge_multi_head_rows(heads)
            n_lines += len(merged)
            for (_b, ct, _cs), (_mb, mt, _ms) in zip(cyr, merged):
                if mt != ct:
                    n_changed += 1
                    n_garbled += is_garbled_greek(mt)
                    print(f"  p{page} REPL {ct!r} -> {mt!r}")
            for _b, t, _s in merged[len(cyr):]:
                n_garbled += is_garbled_greek(t)
                print(f"  p{page} ADD  {t!r}")
    print(f"{tag}: lines={n_lines} replaced={n_changed} garbled_greek_in_output={n_garbled}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", metavar="PDF")
    parser.add_argument("--tag")
    parser.add_argument("--pages", default="1")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--replay", metavar="TAG")
    args = parser.parse_args(argv)
    if args.capture:
        capture(args.capture, args.tag or Path(args.capture).stem, [int(p) for p in args.pages.split(",")], args.dpi)
    failed = run_unit_cases()
    if args.replay:
        replay(args.replay)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
