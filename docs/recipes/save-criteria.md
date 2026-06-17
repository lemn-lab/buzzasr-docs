# Save criteria

In-training "best checkpoint" selection. Three different criteria are used across the 102 languages, chosen empirically.

## Group A: save-by-WER

Pure greedy WER tracking. Save iff WER decreases. Also caps training at `MAX_STEPS=3000`.

**Used by 17 langs:**

```
arabic, catalan, italian, polish, russian, german, spanish,
tamil, latvian, uzbek, cantonese, pashto, swahili, tajik,
georgian, ukrainian, galician
```

Why these specific langs: the rep-trap collapse pattern (loss decreasing while WER explodes) was observed during training and the team patched these templates to be rep-trap-safe.

## Group B: combined-gate

Save iff (`wer < best_wer` OR `loss < best_loss`) AND neither has drifted more than its tolerance above best.

```python
WER_TOL  = 0.05    # WER may drift up to 5 percentage points above best
LOSS_EPS = 0.05    # loss may drift up to 0.05 above best
improved_any   = (wer < best_wer) or (eval_loss < best_loss)
neither_diverged = (wer < best_wer + WER_TOL) and (eval_loss < best_loss + LOSS_EPS)
if improved_any and neither_diverged:
    save_ckpt()
```

**Used by 82 langs** (everything not in Group A or C).

Why: combines the strengths of save-by-WER (gets the best-by-end-task) and save-by-LOSS (smoother signal on small val sets) while guarding against rep-trap drift in either metric.

## Group C: save-by-LOSS

```python
if eval_loss < best_loss:
    save_ckpt()
```

**Used by 4 langs**: burmese, khmer, thai, ganda.

Why: scripts where WER is unreliable as a metric (no-whitespace boundaries, abugida scripts where WER doesn't capture transcription quality).

## See also

- [Pipeline → Eval + checkpoint mechanics](../pipeline/eval-checkpoint.md) — implementation details
- [Recipes → Full fine-tuning (FFT)](fft.md)
