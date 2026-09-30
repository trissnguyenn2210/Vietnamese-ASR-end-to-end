"""Fine-tune PhoWhisper-small on FLEURS Vietnamese."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cache import configure_cache

configure_cache()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jiwer import cer, wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from data import DATASET_CONFIG, SAMPLE_RATE, load_fleurs
from preprocessing import (
    audio_to_mono_16k,
    normalize_for_metrics,
    normalize_transcript,
)

DEFAULT_MODEL = "vinai/PhoWhisper-small"
MAX_AUDIO_SECONDS = 30.0
MIN_AUDIO_SECONDS = 0.3


@dataclass
class WhisperBatchCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _valid_duration(example: dict[str, Any]) -> bool:
    sample_count = example.get("num_samples")
    if sample_count is None:
        return True
    duration = int(sample_count) / SAMPLE_RATE
    return MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS


def _encode_batch(batch: dict[str, list[Any]], processor: WhisperProcessor) -> dict[str, Any]:
    input_features: list[np.ndarray] = []
    labels: list[list[int]] = []
    for audio, transcript in zip(batch["audio"], batch["transcription"]):
        waveform = audio_to_mono_16k(audio)
        duration = len(waveform) / SAMPLE_RATE
        if not MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS:
            raise ValueError(f"Audio duration out of supported range: {duration:.2f}s")
        features = processor.feature_extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="np"
        ).input_features[0]
        input_features.append(features)
        labels.append(
            processor.tokenizer(
                normalize_transcript(transcript), max_length=448, truncation=True
            ).input_ids
        )
    return {"input_features": input_features, "labels": labels}


def _prepare_split(
    split: Any,
    *, processor: WhisperProcessor, limit: int, workers: int, split_name: str,
) -> Any:
    split = split.filter(_valid_duration, desc=f"Filtering {split_name} durations")
    if limit > 0:
        split = split.select(range(min(limit, len(split))))
    return split.map(
        lambda batch: _encode_batch(batch, processor),
        batched=True,
        batch_size=16,
        num_proc=workers,
        remove_columns=split.column_names,
        desc=f"Extracting Whisper features for {split_name}",
    )


def _metric_function(processor: WhisperProcessor):
    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predicted_text = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        reference_text = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        references = [normalize_for_metrics(text) for text in reference_text]
        predictions = [normalize_for_metrics(text) for text in predicted_text]
        return {"wer": float(wer(references, predictions)), "cer": float(cer(references, predictions))}

    return compute_metrics


def _save_training_visuals(history: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "trainer_log_history.csv", index=False)
    if frame.empty or "step" not in frame:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, label, marker in (
        ("loss", "train loss", None),
        ("eval_loss", "validation loss", "o"),
    ):
        if name in frame:
            rows = frame.dropna(subset=[name])
            axes[0].plot(rows["step"], rows[name], label=label, marker=marker)
    axes[0].set(title="Training and validation loss", xlabel="Step", ylabel="Loss")
    axes[0].grid(alpha=0.2)
    if axes[0].lines:
        axes[0].legend()

    for name, label in (("eval_wer", "validation WER"), ("eval_cer", "validation CER")):
        if name in frame:
            rows = frame.dropna(subset=[name])
            axes[1].plot(rows["step"], rows[name], marker="o", label=label)
    axes[1].set(
        title="Validation recognition error", xlabel="Step", ylabel="Error rate (lower is better)"
    )
    axes[1].grid(alpha=0.2)
    if axes[1].lines:
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_CONFIG} from google/fleurs")
    dataset = load_fleurs(cache_dir=args.cache_dir)
    processor = WhisperProcessor.from_pretrained(
        args.model, language="Vietnamese", task="transcribe"
    )
    train_dataset = _prepare_split(
        dataset["train"], processor=processor, limit=args.max_train_examples,
        workers=args.preprocessing_workers, split_name="train",
    )
    validation_dataset = _prepare_split(
        dataset["validation"], processor=processor, limit=args.max_validation_examples,
        workers=args.preprocessing_workers, split_name="validation",
    )
    test_dataset = _prepare_split(
        dataset["test"], processor=processor, limit=args.max_test_examples,
        workers=args.preprocessing_workers, split_name="test",
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "Vietnamese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_arg_values: dict[str, Any] = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_arg_values[strategy_name] = "epoch"
    training_args = Seq2SeqTrainingArguments(**training_arg_values)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=WhisperBatchCollator(
            processor, model.config.decoder_start_token_id
        ),
        compute_metrics=_metric_function(processor),
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
        max_length=225,
        num_beams=1,
    )
    trainer.save_metrics("test", test_metrics)
    _save_training_visuals(trainer.state.log_history, args.output_dir)

    result = {"train": train_result.metrics, "test": test_metrics}
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    print(f"Saved model and training plots to {args.output_dir.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pho-whisper-small-fleurs-vi"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
