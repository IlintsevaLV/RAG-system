"""Core package."""

from core.config import Settings, get_settings
from core.ir import BlockType, BBox, Citation, Provenance, QualitySignals, RegionBlock

__all__ = [
    "Settings",
    "get_settings",
    "BlockType",
    "BBox",
    "Citation",
    "Provenance",
    "QualitySignals",
    "RegionBlock",
]
