# BuzzASR

> A swarm of 100+ monolingual Whisper-based ASR models for FLEURS languages.

## What this is

BuzzASR is a collection of language-specialized fine-tuned Whisper-large-v3 models adapted for automatic speech recognition (ASR) in 102 languages. We use two adaptation strategies:

- **Simple fine-tuning (SFT)** — straightforward Whisper fine-tuning on monolingual speech data
- **Full fine-tuning (FFT)** — a 3-stage pipeline involving per-language tokenizer replacement, multitask fine-tuning on text-only and speech recognition

For 89 of 102 languages, the best BuzzASR model outperforms Whisper-large-v3 zero-shot. On 29 of 102 languages we achieve state-of-the-art CER among open-source systems on the FLEURS+CV25 combined test set.

## What this documentation covers

This site documents *how the code actually works* — not just what the paper says. The goal is that a researcher new to the project (or your future self) can:

- Reproduce paper results from scratch
- Understand the training pipeline end-to-end
- Modify recipes safely
- Diagnose failures
- Extend to new languages

## Where to start

| If you want to… | Read |
|---|---|
| Reproduce a single FFT run | [Getting Started → Single FFT job](getting-started/single-fft-job.md) |
| Understand the orchestration | [Pipeline → Overview](pipeline/overview.md) |
| Modify the tokenizer recipe | [Tokenizers → Recipe](tokenizers/recipe.md) |
| Understand the tokenizer fix | [Tokenizers → Split bug and fix](tokenizers/split-bug-and-fix.md) |
| Pick a hyperparameter config for a new language | [Recipes → Hyperparameter selection](recipes/hyperparameters.md) |

## Project layout (where the code lives)

```
/mnt/ssd-3/asr/
├── _repro_sync/                      # main codebase
│   ├── train_fft.py                  # FFT orchestrator (CLI)
│   ├── train_vft.py                  # VFT (simple FT) orchestrator
│   ├── dispatcher.py                 # multi-GPU job scheduler
│   ├── matrix.json                   # experiment matrix
│   ├── eval_strategy_c_test_combined.py   # test eval for FFT
│   ├── scaling/
│   │   └── train_bpes_bulk.py        # train all 102 tokenizers
│   └── results/
│       ├── tidy_results.csv          # final paper numbers
│       └── matrix_test_fft_*.json    # per-lang FFT test eval
│
├── _pod_share/
│   ├── adapt-env/                    # Python env (tokenizers, transformers, torch)
│   └── jobs/                         # per-lang training templates
│       └── train_strategy_c_<lang>.py  # one per language
│
└── /mnt/ssd-3/checkpoints/
    ├── frankenstein/                 # OLD per-lang BPE tokenizers (pre-fix)
    ├── frankenstein_fix/             # NEW per-lang BPE tokenizers (post-Regex() fix)
    └── matrix_runs/                  # all fine-tuned model checkpoints
```

## Recent updates

!!! note "Tokenizer pre-tokenizer fix"
    The original per-language BPE tokenizers were built with a `Split(pattern=str, ...)` call that HF tokenizers treats as a literal substring match, not a regex. This made the Split pre-tokenizer a no-op for all 102 languages. See [Tokenizers → Split bug and fix](tokenizers/split-bug-and-fix.md) for the full story, audit data, and remediation.


## Citation

```bibtex
@misc{buzzasr2026,
  title  = {BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous ACL submission}
}
```
