"""Configure Hugging Face caches outside the source repository."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CACHE_ROOT = Path(
    os.environ.get("SPEECH_CACHE_DIR", Path.home() / "data" / "fleurs")
).expanduser()


def configure_cache(cache_dir: str | Path | None = None) -> Path:
    """Set Hugging Face cache paths and return the datasets cache directory."""
    cache_root = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_ROOT
    hub_cache = cache_root / "hub"
    datasets_cache = cache_root / "datasets"
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    try:
        import huggingface_hub.constants as hub_constants
    except ImportError:
        pass
    else:
        hub_constants.HF_HUB_CACHE = hub_cache
    return datasets_cache


configure_cache()
