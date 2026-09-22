"""Persistent UniMERNet worker.

Runs inside .venv-unimernet. Reads JSON requests from stdin line by line,
writes JSON responses to stdout line by line.

Request:  {"image_b64": "<base64 PNG>"}
Response: {"latex": "...", "confidence": 0.85}
       or {"error": "..."}

Also emits {"ready": true} once at startup.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _load_model(config_path: str):
    import torch
    from unimernet.common.config import Config
    import unimernet.tasks as tasks
    from unimernet.processors import load_processor

    args = argparse.Namespace(cfg_path=config_path, options=None)
    cfg = Config(args)
    task = tasks.setup_task(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = task.build_model(cfg).to(device)
    model.eval()
    vis_processor = load_processor(
        "formula_image_eval",
        cfg.config.datasets.formula_rec_eval.vis_processor.eval,
    )
    return model, vis_processor, device


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not config_path:
        _emit({"ready": False, "error": "no config_path"})
        return 1

    try:
        model, vis_processor, device = _load_model(config_path)
    except Exception as exc:  # noqa: BLE001
        _emit({"ready": False, "error": f"{exc}\n{traceback.format_exc()}"})
        return 1

    _emit({"ready": True})
    _log_stderr("worker ready")

    import torch
    from PIL import Image

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            b64 = req["image_b64"]
            png = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(png)).convert("RGB")

            tensor = vis_processor(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model.generate({"image": tensor})

            if isinstance(output, dict):
                preds = output.get("pred_str") or []
                latex = preds[0] if preds else ""
            else:
                latex = str(output)

            _emit({"latex": str(latex or ""), "confidence": 0.85})
        except Exception as exc:  # noqa: BLE001
            _emit({"error": f"{exc}", "traceback": traceback.format_exc()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())