"""Prepare RapidOCR ONNX models for offline use (work PC without ModelScope).

Run on a machine with internet (or where rapidocr already cached models):

  python -m scripts.prepare_ocr_models
  python -m scripts.prepare_ocr_models --zip

Then copy data/models/ocr/ (or the zip) to the work PC into the same path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.ocr_rapid import (
    ALL_KNOWN_MODELS,
    OPTIONAL_GREEK_MODEL,
    OPTIONAL_LATIN_MODEL,
    REQUIRED_MODELS,
    check_local_models,
    default_model_dir,
)


def _copy_from_rapidocr_package(dest: Path) -> list[str]:
    try:
        import rapidocr
    except ImportError:
        return list(REQUIRED_MODELS)

    src_dir = Path(rapidocr.__file__).parent / "models"
    copied = []
    missing = []
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_MODELS:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            copied.append(name)
        else:
            missing.append(name)
    print(f"Copied from {src_dir}: {copied}")
    return missing


def _download_modelscope_style(dest: Path, missing: list[str]) -> list[str]:
    """Best-effort download via RapidOCR init (needs ModelScope access)."""
    if not missing:
        return []
    try:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        print("Trying RapidOCR auto-download (ModelScope)...")
        RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.CYRILLIC,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Global.model_root_dir": str(dest.resolve()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Auto-download failed: {exc}")

    # copy again from package cache if download landed there
    return _copy_from_rapidocr_package(dest)



def _copy_from_rapidocr_package_named(dest: Path, names: list[str]) -> list[str]:
    try:
        import rapidocr
    except ImportError:
        return list(names)
    src_dir = Path(rapidocr.__file__).parent / "models"
    missing = []
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
        else:
            missing.append(name)
    return missing


def _download_optional(dest: Path, filename: str, url_suffix: str, label: str) -> None:
    url = (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
        + url_suffix
    )
    out = dest / filename
    try:
        import urllib.request

        print(f"Downloading {label} OCR model:", url)
        urllib.request.urlretrieve(url, out)
        print(f"{label} model saved:", out, out.stat().st_size)
    except Exception as exc:  # noqa: BLE001
        print(f"{label} model download failed: {exc}")


def _download_optional(dest: Path, filename: str, url_suffix: str, label: str) -> None:
    url = (
        "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/"
        + url_suffix
    )
    out = dest / filename
    try:
        import urllib.request

        print(f"Downloading {label} OCR model:", url)
        urllib.request.urlretrieve(url, out)
        print(f"{label} model saved:", out, out.stat().st_size)
    except Exception as exc:  # noqa: BLE001
        print(f"{label} model download failed: {exc}")


def _download_greek(dest: Path) -> None:
    _download_optional(
        dest,
        OPTIONAL_GREEK_MODEL,
        "onnx/PP-OCRv5/rec/el_PP-OCRv5_rec_mobile.onnx",
        "Greek",
    )


def _download_latin(dest: Path) -> None:
    _download_optional(
        dest,
        OPTIONAL_LATIN_MODEL,
        "onnx/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
        "Latin",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare offline RapidOCR models")
    parser.add_argument("--out", default=None, help="Target dir (default data/models/ocr)")
    parser.add_argument("--zip", action="store_true", help="Also write ocr_models.zip")
    args = parser.parse_args(argv)

    dest = Path(args.out) if args.out else default_model_dir()
    dest.mkdir(parents=True, exist_ok=True)

    missing = check_local_models(dest)
    if missing:
        print(f"Missing in {dest}: {missing}")
        missing = _copy_from_rapidocr_package(dest)
    if missing:
        missing = _download_modelscope_style(dest, missing)

    still = check_local_models(dest)
    if still:
        print("FAILED. Still missing:", still)
        print(
            "On a PC with ModelScope access run this script, then copy the folder:\n"
            f"  {dest.resolve()}\n"
            "to the work PC (USB / shared drive). Do not rely on git for .onnx files."
        )
        return 1

    # Optional heads: Latin (English) + Greek (formulas / notations).
    for filename, downloader, label in (
        (OPTIONAL_LATIN_MODEL, _download_latin, "latin"),
        (OPTIONAL_GREEK_MODEL, _download_greek, "greek"),
    ):
        path = dest / filename
        if not path.is_file():
            print(f"Optional {label} model missing: {filename}")
            missing_opt = _copy_from_rapidocr_package_named(dest, [filename])
            if missing_opt:
                downloader(dest)

    print(f"OK. Models ready in {dest.resolve()}")
    for name in REQUIRED_MODELS:
        size = (dest / name).stat().st_size
        print(f"  {name}  ({size / 1e6:.1f} MB)")
    for filename, label in (
        (OPTIONAL_LATIN_MODEL, "latin/english"),
        (OPTIONAL_GREEK_MODEL, "greek"),
    ):
        path = dest / filename
        if path.is_file():
            print(f"  {filename}  ({path.stat().st_size / 1e6:.1f} MB)  [{label}]")
        else:
            print(f"  WARN: {filename} not installed — {label} may be weak")

    if args.zip:
        zip_path = dest.parent / "ocr_models.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in ALL_KNOWN_MODELS:
                src = dest / name
                if src.is_file():
                    zf.write(src, arcname=f"ocr/{name}")
        print(f"Zip: {zip_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
