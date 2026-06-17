# 102-language tokfix retrain

The project-wide FFT retrain using the [tokenizer fix](../tokenizers/split-bug-and-fix.md) on all 102 BuzzASR languages.

## Status

!!! info "In progress"
    Launched 2026-06-17. Running on 16 GPUs (8 per pod × 2 pods). Wall-clock ETA: ~14-20h from launch.

## Setup

- **Tokenizers**: all 102 retrained with `Split(pattern=Regex(SPLIT_PATTERN), ...)` → `/mnt/ssd-3/checkpoints/frankenstein_fix/`
- **Templates**: aligned to deployed code per-language save criterion (Group A / B / C)
- **Recipe**: paper-matched per-lang cfg (WER<30% → cfg B, 30-70% → cfg D, >70% → cfg A; with 4 empirical exceptions)
- **Seed**: 2024
- **Training data**: FLEURS-train + CV-train combined IID (unchanged from paper)
- **Dispatcher**: filtered to `_tokfix` matrix entries only, on both shivam-0 and shivam-2

## Job naming convention

```
fft_<lang>_cfg<X>_s2024_main_tokfix
```

The `_tokfix` suffix distinguishes these from the original paper-era main FFT runs.

## What we expect from this run

Per the [5-lang ablation](ptfix-5lang-ablation.md):

- **Japanese**: large CER drop (-10 pts). Resolves the documented Section C limitation.
- **Brahmic / Abugida (17 langs)**: largest tokenizer improvements (chars/tok dropped 75%). Likely the biggest WER wins.
- **CJK, Hebrew, Armenian, Georgian, Ethiopic**: moderate improvements (chars/tok dropped 25–30%).
- **Cyrillic, Arabic-script**: small improvements (chars/tok dropped 18–20%).
- **Latin (62 langs)**: tokenization was already mostly fine; expect small ± 1-2 point WER variance.

## Plan after retrain completes

1. **Test eval all 102 ckpts** on FLEURS-test + CV25-test + combined (`eval_strategy_c_test_combined.py`, ~half a day)
2. **Re-measure decoder latency** with the fixed tokenizers (resolves the Latin-script speedup paradox in Appendix Table 3)
3. **Regenerate `tidy_results.csv`** with new FFT numbers alongside the existing SFT + Whisper-zs + external baselines
4. **Re-render paper plots** (Figure 1 scatter, Figure 4 KDE)

## See also

- [5-lang ptfix ablation](ptfix-5lang-ablation.md) — the small experiment that validated the approach
- [Tokenizers → Split bug and fix](../tokenizers/split-bug-and-fix.md) — the bug being fixed
- [Recipes → Save criteria](../recipes/save-criteria.md) — Group A/B/C save logic
