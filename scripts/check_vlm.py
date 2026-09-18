"""Check that llama-server VLM is reachable the way the pipeline expects.

Usage (work PC, venv active):
  python -m scripts.check_vlm
  python -m scripts.check_vlm --ping-image data/cache/extract/.../page_0246_cd.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.vlm_client import LlamaVisionClient, VLM_PAGE_READ_PROMPT


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Smoke-check local VLM (llama-server)")
    parser.add_argument("--host", default=settings.model_host)
    parser.add_argument("--port", type=int, default=settings.model_port)
    parser.add_argument(
        "--ping-image",
        default=None,
        help="Optional page/crop image to run a short extract prompt",
    )
    args = parser.parse_args(argv)

    client = LlamaVisionClient(host=args.host, port=args.port, timeout_s=60.0)
    print(f"Checking http://{args.host}:{args.port} ...")
    if not client.ready():
        print("FAIL: llama-server not reachable (/health or /v1/models).")
        print("Start it first (see scripts/start_vlm.ps1), then retry.")
        return 1
    print("OK: server is up.")

    if args.ping_image:
        path = Path(args.ping_image)
        if not path.is_file():
            print(f"FAIL: image not found: {path}")
            return 1
        print(f"Sending image: {path}")
        try:
            reply = client.read_image(str(path), VLM_PAGE_READ_PROMPT)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: chat/completions error: {exc}")
            return 1
        preview = (reply or "").strip().replace("\n", " ")
        print(f"OK: got {len(reply or '')} chars")
        print(f"preview: {preview[:300]!r}")
        if (reply or "").strip().upper().startswith("UNREADABLE"):
            print("NOTE: model returned UNREADABLE — page may be bad or model too weak.")
    else:
        print("Tip: re-run with --ping-image <png> to test vision path.")

    print("\nNext:")
    print("  set ENABLE_VLM=true in .env  (or use --enable-vlm on CLI)")
    print(
        "  python -m ingestion.run_extract_pages data\\raw\\US_Army_...pdf "
        "--pages 83,246 --enable-vlm"
    )
    print(
        "  python -m ingestion.run_regions data\\raw\\US_Army_...pdf "
        "--pages 246 --enable-vlm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
