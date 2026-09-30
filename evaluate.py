"""Evaluate a PhoWhisper checkpoint on one official FLEURS split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from cache import configure_cache

configure_cache()

import torch
from jiwer import cer, wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data import SAMPLE_RATE, SPLITS, load_fleurs
from preprocessing import (
    audio_to_mono_16k,
    normalize_for_metrics,
    normalize_transcript,
)
from train import DEFAULT_MODEL


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_fleurs(cache_dir=args.cache_dir)
    split = dataset[args.split]
    if args.max_examples > 0:
        split = split.select(range(min(args.max_examples, len(split))))
    processor = WhisperProcessor.from_pretrained(
        args.checkpoint, language="Vietnamese", task="transcribe"
    )
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=dtype
    )
    model.generation_config.language = "Vietnamese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    device = "cuda" if use_cuda else "cpu"
    model.to(device)
    model.eval()

    rows: list[dict[str, str]] = []
    for start in range(0, len(split), args.batch_size):
        end = min(start + args.batch_size, len(split))
        batch_rows = [split[index] for index in range(start, end)]
        waveforms = [audio_to_mono_16k(row["audio"]) for row in batch_rows]
        inputs = processor.feature_extractor(
            waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)
        if use_cuda:
            input_features = input_features.to(dtype=dtype)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features, max_new_tokens=225, num_beams=args.num_beams
            )
        predictions = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        for row, prediction in zip(batch_rows, predictions):
            rows.append(
                {
                    "id": str(row.get("id", "")),
                    "reference": normalize_transcript(row["transcription"]),
                    "prediction": normalize_transcript(prediction),
                }
            )
        if len(rows) % (args.batch_size * 10) == 0 or end == len(split):
            print(f"Evaluated {len(rows)}/{len(split)} examples")

    references = [normalize_for_metrics(row["reference"]) for row in rows]
    predictions = [normalize_for_metrics(row["prediction"]) for row in rows]
    metrics = {
        "checkpoint": args.checkpoint,
        "dataset": "google/fleurs",
        "config": "vi_vn",
        "split": args.split,
        "examples": len(rows),
        "wer_syllable": float(wer(references, predictions)),
        "cer": float(cer(references, predictions)),
        "metric_normalization": "NFC, lowercase, punctuation removed; Vietnamese diacritics retained",
    }
    output_dir = args.output_dir or Path("artifacts") / (
        f"eval-{Path(args.checkpoint).name}-{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics and predictions to {output_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
