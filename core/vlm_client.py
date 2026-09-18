"""Vision LLM client for llama.cpp / OpenAI-compatible /v1/chat/completions."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import httpx
import numpy as np

from core.interfaces import VisionClient


def encode_image_bgr_jpeg(image_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class LlamaVisionClient(VisionClient):
    """Calls OpenAI-compatible multimodal chat API (llama-server)."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        model: str = "vlm",
        timeout_s: float = 180.0,
    ) -> None:
        self.base = f"http://{host}:{port}"
        self.model = model
        self.timeout_s = timeout_s

    def ready(self) -> bool:
        try:
            r = httpx.get(f"{self.base}/health", timeout=3.0)
            return r.status_code < 500
        except Exception:
            try:
                r = httpx.get(f"{self.base}/v1/models", timeout=3.0)
                return r.status_code < 500
            except Exception:
                return False

    def read_image(self, image_path: str, prompt: str) -> str:
        data = Path(image_path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return self._chat_image_b64(b64, prompt, mime="image/png")

    def read_ndarray(self, image_bgr: np.ndarray, prompt: str) -> str:
        b64 = encode_image_bgr_jpeg(image_bgr)
        return self._chat_image_b64(b64, prompt, mime="image/jpeg")

    def _chat_image_b64(self, b64: str, prompt: str, *, mime: str) -> str:
        url = f"{self.base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


# Prompts
LAYOUT_CHECK_PROMPT = """You are checking a technical document page image.
The text below was extracted from the PDF text layer (may be incomplete or wrong order).

Extracted text (truncated):
---
{extracted}
---

Tasks (be brief, structured):
1) How many text columns? (1 or 2 or other)
2) Is there a table? (yes/no)
3) Unusual layout? (rotated text, side notes, multi-region) yes/no + short note
4) Does extracted text miss major readable regions? yes/no
5) If columns=2, confirm whether reading order should be left-column-then-right.

Reply in plain text lines starting with: COLUMNS: TABLE: UNUSUAL: MISSING: ORDER:
"""

VLM_PAGE_READ_PROMPT = """Extract all readable body text from this technical document page image.
Rules:
- Preserve reading order (if two columns: finish left column, then right).
- Keep section numbers and headings.
- For tables: output row-wise plain text with | separators; do not invent cells.
- Skip decorative lines; keep captions under figures if readable.
- If the page is mostly unreadable, write exactly: UNREADABLE
- Output plain text only, no markdown fences.
"""
