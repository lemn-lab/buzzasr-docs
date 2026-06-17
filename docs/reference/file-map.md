# File map

Quick reference: where each piece of the codebase lives.

## Top-level

```
/mnt/ssd-3/asr/
├── _repro_sync/                       # main codebase
├── _pod_share/                        # shared across pods
└── (no other top-level dirs)
```

## `_repro_sync/`

```
_repro_sync/
├── train_fft.py                       # FFT orchestrator (CLI entry)
├── train_vft.py                       # VFT/SFT orchestrator (CLI entry)
├── dispatcher.py                      # multi-GPU job scheduler
├── eval_strategy_c_test_combined.py   # FFT test eval
├── eval_vft_test.py                   # VFT/SFT test eval
├── run_test_eval.py                   # batch test eval driver
├── matrix.json                        # experiment manifest
├── matrix_status.json                 # per-job state
├── matrix_tokfix_only.json            # filtered subset (tokenizer-fix retrain)
│
├── scaling/
│   ├── train_bpes_bulk.py             # train all 102 per-lang tokenizers
│   └── train_bpe_nopretoken.py        # ablation: BPE with no Split pre-tokenizer
│
├── results/
│   ├── tidy_results.csv               # final results table
│   ├── matrix_test_fft_*.json         # per-lang FFT test eval
│   └── ptfix/                         # 5-lang ablation results
│
├── logs/
│   ├── matrix/<job_id>.log            # per-job training log
│   └── dispatcher_<pod>.log           # per-dispatcher log
│
└── run_state/
    ├── claims.lock                    # fcntl lock file for dispatcher claims
    └── fft_scripts/                   # generated per-job training scripts
        └── train_fft_<lang>_cfg<X>_s<N>_main.py
```

## `_pod_share/`

```
_pod_share/
├── adapt-env/                         # Python venv (torch, transformers, etc.)
└── jobs/                              # per-lang FFT training templates
    └── train_strategy_c_<lang>.py     # one per language
```

## Checkpoints

```
/mnt/ssd-3/checkpoints/
├── frankenstein/                      # OLD per-lang BPE tokenizers (broken Split)
│   └── <lang>/tokenizer/
│       ├── tokenizer.json
│       ├── vocab.json
│       └── merges.txt
│
├── frankenstein_fix/                  # NEW per-lang BPE tokenizers (Regex() fixed)
│   └── <lang>/tokenizer/...
│
└── matrix_runs/                       # all fine-tuned model checkpoints
    └── <job_id>/
        ├── best/checkpoint.pt
        └── latest/checkpoint.pt
```

## Data

```
/mnt/ssd-3/asr/
├── fleurs/                            # FLEURS audio + transcripts
│   └── data/<fleurs_code>/
│
├── cv/cv-corpus-25.0-2026-03-09/      # CommonVoice v25
│   └── <cv_code>/clips/ + test.tsv
│
├── training_sets/<fleurs_code>/       # FLEURS+CV combined IID (pre-built)
│
└── text_pretraining/goldfish/         # Per-lang text corpora (for tokenizer + text-MTL)
    └── <iso3>_<script>.txt
```

## See also

- [Pipeline → Overview](../pipeline/overview.md) — how these pieces connect at runtime
- [Reference → Glossary](glossary.md) — what the file/dir names mean
