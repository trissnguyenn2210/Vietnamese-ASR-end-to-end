# Vietnamese ASR end-to-end

This repository independently fine-tunes VinAI PhoWhisper-small on Google
FLEURS Vietnamese (`vi_vn`), evaluates checkpoints, and transcribes local audio.
It includes its own dataset loader and preprocessing helpers and does not depend
on the analysis repository.

## Setup

Python 3.11 is pinned in `.python-version`.

```bash
uv sync
```

The dataset cache and Hugging Face model cache default to `~/data/fleurs/`.
Set `SPEECH_CACHE_DIR` before running a command to change the cache root.
Training outputs default to the ignored `artifacts/` folder in this repository.

## Run

Evaluate the pretrained model on validation:

```bash
uv run speech-evaluate --checkpoint vinai/PhoWhisper-small --split validation
```

Fine-tune and evaluate on the held-out test split:

```bash
uv run speech-train --epochs 3
```

Transcribe a local audio file:

```bash
uv run speech-transcribe ./sample.wav \
  --checkpoint artifacts/pho-whisper-small-fleurs-vi
```

The training pipeline filters clips outside 0.3–30 seconds, converts audio to
mono 16 kHz, keeps Vietnamese diacritics, chooses the checkpoint by validation
WER, and evaluates the test split once. Vietnamese WER is whitespace-tokenized;
CER is reported alongside it.

## Results

The included `results/` files record the existing run. Fine-tuned test results
were WER 8.68% and CER 4.72%; the pretrained validation baseline was WER 11.32%
and CER 7.33%. The existing 6.3 GB checkpoint and full local artifacts are
preserved at `~/data/speech-artifacts/pho-whisper-small-fleurs-vi/` and are not
uploaded to GitHub. To use that checkpoint:

```bash
uv run speech-transcribe ./sample.wav \
  --checkpoint ~/data/speech-artifacts/pho-whisper-small-fleurs-vi
```

New training runs create a checkpoint under this repository's ignored
`artifacts/` folder.

The dataset is [Google FLEURS](https://huggingface.co/datasets/google/fleurs),
configuration `vi_vn`, licensed CC-BY 4.0. Retain attribution when sharing
derived work.
