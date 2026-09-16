"""Smoke checks for the work PC before heavy pipeline stages."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# allow running as script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_settings
from core.logger import get_logger, setup_logging


def _check_nvidia() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "detail": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
        return {"ok": True, "detail": out}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}


def _check_http(url: str) -> dict:
    try:
        import httpx

        r = httpx.get(url, timeout=3.0)
        return {"ok": r.status_code < 500, "detail": f"status={r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    log = get_logger("smoke")
    settings.ensure_dirs()

    checks = {
        "normativka_dir": {
            "ok": settings.normativka_dir.exists(),
            "detail": str(settings.normativka_dir.resolve()),
        },
        "gpu": _check_nvidia(),
        "qdrant": _check_http(f"{settings.qdrant_url.rstrip('/')}/readyz"),
        "neo4j_bolt_port": {
            "ok": True,
            "detail": f"configured {settings.neo4j_uri} (TCP probe skipped)",
        },
        "vlm_flag": {
            "ok": True,
            "detail": f"ENABLE_VLM={settings.enable_vlm}",
        },
    }

    failed = False
    for name, result in checks.items():
        status = "OK" if result["ok"] else "FAIL"
        if not result["ok"] and name in {"normativka_dir"}:
            failed = True
        log.info("check", name=name, status=status, detail=result["detail"])

    if failed:
        log.error("smoke_failed")
        return 1
    log.info("smoke_done", hint="start docker compose if qdrant/neo4j FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
