"""Transcribe a local audio file with PhoWhisper."""

from __future__ import annotations

import argparse
from pathlib import Path

from cache import configure_cache

configure_cache()

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from data import SAMPLE_RATE
from preprocessing import audio_to_mono_16k
from train import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--stride-length-s", type=int, default=5)
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    waveform = audio_to_mono_16k({"array": waveform, "sampling_rate": sample_rate})
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

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if use_cuda else -1,
        torch_dtype=dtype,
    )
    result = asr(
        {"array": waveform, "sampling_rate": SAMPLE_RATE},
        chunk_length_s=args.chunk_length_s,
        stride_length_s=args.stride_length_s,
        generate_kwargs={
            "language": "Vietnamese", "task": "transcribe", "num_beams": 1
        },
    )
    transcription = result["text"].strip()
    print(transcription)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(transcription + "\n", encoding="utf-8")
        print(f"Saved transcript to {args.output.resolve()}")


if __name__ == "__main__":
    main()
