# Installation

!!! note "Placeholder — fill in as code-release plan firms up"
    This page should cover environment setup once the code-release path is decided.

## Hardware requirements

- **GPU**: 1× A100 80GB or H100 (FFT runs require ~38 GB VRAM for Whisper-large-v3 at `batch_size=2, grad_accum=16`)
- **Disk**: ~6 GB per checkpoint × ~300 ckpts = ~2 TB for the full matrix
- **RAM**: ≥64 GB recommended (goldfish corpora can be 10 GB)

## Python environment

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate

# Core deps
pip install \
    torch \
    transformers \
    datasets \
    tokenizers \
    jiwer \
    librosa \
    soundfile \
    regex \
    wandb
```

## External data

- **FLEURS**: pulled from `google/fleurs` via HuggingFace `datasets`
- **CommonVoice v25**: download from <https://commonvoice.mozilla.org/datasets>
- **Goldfish text corpora**: download from <https://github.com/tylerachang/goldfish>

## Repository checkout

(To fill in once the code is released publicly.)

## See also

- [Single FFT job](single-fft-job.md) — for running a single fine-tune end-to-end
- [Reproducing paper results](reproducing-paper-results.md) — for the full pipeline
