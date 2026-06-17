# Hyperparameter selection

!!! note "Placeholder — write up the per-lang cfg selection rule"

## The four configs

| cfg | Embed LR mult | Decoder LR mult | When used |
|---|---:|---:|---|
| A | 3.0× | 3.0× | Aggressive — for langs where Whisper has a weak prior |
| B | 1.0× | 1.0× | Conservative — for langs where Whisper is near-state-of-the-art |
| C | 3.0× | 1.0× | Embed-heavy — for langs where the vocab change is the main bottleneck |
| D | 2.0× | 2.0× | Middle — between A and B, for scaling matrix langs in the mid-WER range |

## Assignment rule

For 102-lang scaling (post-17-lang pilot):

- **Whisper-zs WER < 30%** → cfg B
- **30% ≤ Whisper-zs WER ≤ 70%** → cfg D
- **Whisper-zs WER > 70%** → cfg A

Per-lang base learning rate η for the 17 core langs is set from the Bayesian sweep (see paper Appendix B.2 Table 2).

## See also

- [Save criteria](save-criteria.md)
- [FFT recipe](fft.md)
