# Reproducing paper results

End-to-end workflow to reproduce the BuzzASR Table 5/6 numbers.

!!! note "Placeholder — fill in once code-release path is finalized"
    This page should walk through cloning the repo, downloading data, training, and evaluating.

## High-level steps

1. **Set up environment** — see [Installation](installation.md)
2. **Download FLEURS** — pulled lazily via `datasets.load_dataset("google/fleurs", lang_code)`
3. **Download CommonVoice v25** — from <https://commonvoice.mozilla.org/datasets>, capped at 90h per lang
4. **Download goldfish corpora** — for the 102 langs we use
5. **Train per-lang tokenizers** — `python scaling/train_bpes_bulk.py` (~10 min on CPU)
6. **Run all 102 FFT jobs** — `python dispatcher.py --methods fft --gpus 0-7` (~3-5 days on 8 GPUs)
7. **Run all SFT jobs** — `python dispatcher.py --methods vft --gpus 0-7` (~half the FFT time)
8. **Test eval all checkpoints** — `python run_test_eval.py --gpus 0-7` (~half a day)
9. **Aggregate to tidy_results.csv** — `python results/build_tidy_results.py`
10. **Generate plots** — see notebook (TBD link)

## Per-step timing on 8× A100 80 GB

| Step | Wall-clock |
|---|---|
| Tokenizer training (102 langs) | ~10 min |
| FFT main runs (102 langs × 2 seeds = 204) | ~3-5 days |
| SFT main runs (102 langs × 2 configs = 204) | ~1.5-2 days |
| Test eval (all ckpts) | ~12 hours |
| External baseline runs (Cohere, Qwen3, MMS, Omni) | varies by API rate-limits |

## Storage requirements

- Tokenizers: ~3 GB total
- Model checkpoints: ~5-6 GB each × ~400 final ckpts ≈ 2.5 TB
- Goldfish corpora: ~150 GB total
- FLEURS audio + CV audio caches: ~80 GB

## See also

- [Pipeline → Overview](../pipeline/overview.md) — orchestration
- [Recipes → Hyperparameter selection](../recipes/hyperparameters.md) — per-lang cfg choices
