"""Dataset access helpers for the Vietnamese FLEURS experiment."""

from __future__ import annotations

from typing import Any

from cache import configure_cache

DATASET_ID = "google/fleurs"
DATASET_CONFIG = "vi_vn"
SAMPLE_RATE = 16_000
SPLITS = ("train", "validation", "test")


def load_fleurs(*, streaming: bool = False, cache_dir: str | None = None) -> Any:
    """Load the official Vietnamese FLEURS splits at 16 kHz."""
    datasets_cache = configure_cache(cache_dir)
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {
        "streaming": streaming,
        "cache_dir": str(datasets_cache),
    }
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, **kwargs)
    for split in SPLITS:
        dataset[split] = dataset[split].cast_column(
            "audio", Audio(sampling_rate=SAMPLE_RATE)
        )
    return dataset
