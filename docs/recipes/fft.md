# Full fine-tuning (FFT)

!!! note "Placeholder — write up FFT recipe details"

## What FFT is

A 3-stage pipeline:

1. **Tokenizer replacement** — swap Whisper's multilingual BPE for a per-lang BPE trained on monolingual text
2. **Multitask fine-tuning** — interleave ASR and text-only batches in a single training stage
3. **Save best-by-criterion** — Group A/B/C per language (see [Save criteria](save-criteria.md))

## When to use FFT

- Non-Latin scripts where Whisper's BPE fragments badly (chars/token < 2)
- Low-resource languages where text-only data is more abundant than ASR data

## See also

- [Simple fine-tuning (SFT)](sft.md)
- [Save criteria](save-criteria.md)
- [Tokenizers → Recipe](../tokenizers/recipe.md)
