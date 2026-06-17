# Running a single FFT job

How to run one full fine-tune end-to-end without the dispatcher / matrix infrastructure.

## Direct invocation

```bash
python train_fft.py \
  --lang spanish \
  --fleurs-code es_419 \
  --whisper-lang spanish \
  --config-id B \
  --seed 1337 \
  --asr-ratio 0.5 \
  --max-text-lines 500000 \
  --text-mtl-path /path/to/text_pretraining/goldfish/spa_latn.txt \
  --output-dir runs/spanish-cfgB-s1337 \
  --init-strategy C
```

What each flag means:

| Flag | Meaning |
|---|---|
| `--lang` | Lang name (must match a `train_strategy_c_<lang>.py` template) |
| `--fleurs-code` | FLEURS dataset code (e.g. `es_419`) |
| `--whisper-lang` | Whisper's language token name (e.g. `spanish`) |
| `--config-id` | Which LR config to use (A=aggressive, B=conservative, C=embed-heavy, D=middle) |
| `--seed` | Random seed |
| `--asr-ratio` | Fraction of batches that are ASR (vs. text-MTL) |
| `--max-text-lines` | Cap on text-MTL data |
| `--text-mtl-path` | Path to per-lang goldfish corpus |
| `--output-dir` | Where to save best/ and latest/ checkpoints |
| `--init-strategy` | `C` for Strategy C (warm-start embedding init) |

## What happens under the hood

`train_fft.py` reads the per-lang template at `_pod_share/jobs/train_strategy_c_<lang>.py`, patches the LR mults / seed / output dir / etc. into it, writes the patched script to `run_state/fft_scripts/`, then exec's it. See [Pipeline → Per-job script generation](../pipeline/per-job-script.md) for the patching details.

## Expected runtime

- A100 80 GB: ~3–4 hours for cfg B (24 grad_accum), ~2.5 hours for cfg A (16 grad_accum)
- H100 80 GB: ~1.5× faster than A100

## Outputs

After training:

```
runs/spanish-cfgB-s1337/
├── best/
│   ├── checkpoint.pt         # ~6 GB FP32 state dict
│   └── training_config.json  # the resolved cfg + best metrics
└── latest/
    └── checkpoint.pt         # periodic backup (every 1000 steps)
```

## Test eval

After training, evaluate on FLEURS-test + CV25-test:

```bash
python eval_strategy_c_test_combined.py \
  --lang spanish \
  --fleurs es_419 \
  --whisper-lang spanish \
  --cv-code es \
  --ckpt-path runs/spanish-cfgB-s1337/best/checkpoint.pt \
  --results-path results/test_spanish_cfgB_s1337.json
```

Output `test_spanish_cfgB_s1337.json` will have `fleurs_test`, `cv25_test`, and `combined` blocks with raw + normalized WER/CER.
