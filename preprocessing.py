"""Small, repeatable audio and transcript preprocessing helpers."""

from __future__ import annotations

import re
import unicodedata
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from data import SAMPLE_RATE

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """Normalize Unicode and whitespace while preserving Vietnamese marks."""
    text = unicodedata.normalize("NFC", str(text))
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_metrics(text: str) -> str:
    """Case and punctuation normalization used consistently for WER/CER."""
    text = normalize_transcript(text).casefold()
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def audio_to_mono_16k(audio: dict[str, Any]) -> np.ndarray:
    """Return finite, mono float audio at 16 kHz without loudness distortion."""
    waveform = np.asarray(audio["array"], dtype=np.float32)
    sample_rate = int(audio["sampling_rate"])

    if waveform.ndim == 2:
        # Hugging Face audio normally uses [samples, channels], but tolerate
        # [channels, samples] files too.
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.T
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Expected mono or stereo audio, got shape {waveform.shape}")

    waveform = np.nan_to_num(waveform, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(sample_rate, SAMPLE_RATE)
        waveform = resample_poly(
            waveform, SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)
