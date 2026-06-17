# 5-language ptfix ablation

A focused experiment to validate the [Split bug fix](../tokenizers/split-bug-and-fix.md) on a small set before committing to a project-wide retrain.

## Setup

- **Languages**: arabic + 4 CJK (Mandarin, Cantonese, Japanese, Korean)
- **Seed**: 2024 (new seed; existing main FFT used 42 + 1337)
- **Config**: paper-matched cfg per Whisper-ZS WER rule
- **Tokenizer**: post-Regex()-fix from `frankenstein_pretokenfix/`
- **Training data**: FLEURS-train + CV-train combined IID (same as main FFT)
- **Recipe**: each lang's deployed save criterion + LR config (no per-lang recipe changes vs paper)

## What this isolated

The point of this ablation was to test: **does the tokenizer fix alone help, holding everything else fixed?**

Specifically, the only differences from the paper's FFT main runs were:

1. Tokenizer source: `frankenstein_pretokenfix/` (Regex()-applied) instead of `frankenstein/` (broken)
2. New seed (2024)

Recipe, training data, eval methodology, and per-lang config all match the paper.

## Results (FLEURS+CV combined test set, normalized)

| Lang | Paper FFT WER | ptfix WER | Δ WER | Paper FFT CER | ptfix CER | Δ CER |
|---|---:|---:|---:|---:|---:|---:|
| **japanese** | 106.94 | 91.87 | **−15.07 ✓** | 50.49 | 40.22 | **−10.27 ✓** |
| arabic | 30.38 | 42.57 | +12.19 | 8.96 | 15.39 | +6.43 |
| cantonese | 43.23 | 54.75 | +11.52 | 30.44 | 32.86 | +2.42 |
| korean | 40.54 | 45.69 | +5.15 | 17.11 | 20.42 | +3.31 |
| mandarin | 43.18 | 61.98 | +18.80 | 23.27 | 41.30 | +18.03 |

**Japanese rescued**: the documented "FFT generation failure" (WER > 100%) in the paper Appendix Table 3 footnote drops to WER 91.87%, CER 40.22%. The root cause was the tokenizer bug.

**Other 4 langs regressed**: this is a real cost of swapping just the tokenizer without changing anything else. With the new vocabulary IDs, the model effectively starts from a different point in embedding space and the existing recipe's hyperparameter choices don't translate one-to-one. We saw this pattern in our analysis: better tokenizer + unchanged recipe ≠ better final WER for these 4 langs.

## What this tells us

1. **The tokenizer bug is real and Japanese-paper-limitation is fixable** by just swapping in the corrected tokenizer.
2. **For the project-wide retrain to monotonically improve**, the recipe needs alignment too — same idea as the team's prior rep-trap recipe work for the 7 fix-equipped templates.

The full 102-lang retrain (in progress) tests both fixes together: corrected tokenizer + per-lang recipe alignment to deployed code.

## See also

- [Tokenizers → Split bug and fix](../tokenizers/split-bug-and-fix.md) — the root cause
- [102-lang tokfix retrain](tokfix-retrain.md) — the full project-wide retrain
