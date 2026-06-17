# The Split bug and the Regex() fix

This page documents a tokenizer pre-tokenizer bug we discovered in late 2026 that affected all 102 BuzzASR tokenizers as originally trained, along with the one-line fix.

## TL;DR

```python
# Before (broken):
Split(pattern=SPLIT_PATTERN, behavior="isolated")

# After (correct):
from tokenizers import Regex
Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated")
```

The HuggingFace `tokenizers` library's `Split` pre-tokenizer accepts both `str` and `Regex` for `pattern`. With a plain `str`, it matches as a **literal substring** — no regex semantics. Our 200-character `SPLIT_PATTERN` was searched as a verbatim substring, never found a match, and the Split silently produced no splits at all. The `Sequence([Split, ByteLevel])` pipeline collapsed to just `ByteLevel(use_regex=False)` — byte-level BPE on raw input, for every language.

## How the bug presents

The intended pipeline:

```python
pre_tokenizer = Sequence([
    Split(pattern=SPLIT_PATTERN, behavior="isolated"),   # script-isolate + word splits
    ByteLevel(add_prefix_space=False, use_regex=False),
])
```

`SPLIT_PATTERN` was a regex containing CJK character classes, GPT-2-style contractions, Unicode-aware `\p{L}+` letter patterns, etc. The intent was that for non-Latin scripts the regex would isolate each character into its own pre-token; for Latin scripts it would do word-level pre-tokenization.

But with `pattern=str` instead of `pattern=Regex(str)`, the Split tried to match the literal 200-character string in the input. That literal never appeared in any natural-language text, so zero matches were found, and `behavior="isolated"` produced no pre-token boundaries. The entire input passed through as a single pre-token.

The `ByteLevel(use_regex=False)` second stage didn't add any further splitting (`use_regex=False` is what it sounds like). So BPE training saw the input as a continuous byte stream and merged freely across script boundaries, word boundaries, punctuation, everything.

## Evidence chain that led to the diagnosis

We didn't find this by reading the source. We found it by chasing symptoms:

### Symptom 1: Kamba's 18× compression cliff

Kamba's pre-training corpus is small (3340 lines, ~700 KB). Its BPE tokenizer reported a compression rate of **74 chars/token on training data** but only **4 chars/token on FLEURS-test**. An 18× cliff — the tokenizer had memorized whole multi-word training fragments as single tokens that didn't fire on novel text.

### Symptom 2: Japanese FFT generation failure

Documented in the paper Appendix C, Table 3 caption: *"The CJK regression is driven by Japanese, where FFT exhibits a generation failure (test WER > 100%)"*. The Japanese model couldn't converge — its loss kept dropping but its WER blew up.

### Symptom 3: Multi-word phrase tokens everywhere

Inspecting tokens emitted for FLEURS-test sentences showed memorized multi-word phrase tokens in every language:

| Lang | Memorized phrase token |
|---|---|
| Spanish | `Ġcon el propósito de Ġ` (one token) |
| Mandarin | `指出："` |
| Japanese | `ことができます。` |
| Cantonese | `Ġ獨立國家` |
| Arabic | (a complete 31-char greeting phrase) |

A proper char-isolation pre-tokenizer should produce CJK tokens of length ~1 char on test text (since each char is its own pre-token, and BPE can only merge within a pre-token). We were seeing chars/token in the 1.6–2.9 range for CJK — far above the 1.0 floor that char isolation would produce.

### The diagnostic that nailed it

```python
from tokenizers.pre_tokenizers import Split

# Should match each Japanese kana/kanji as its own pre-token:
sp = Split(pattern="[぀-ヿ一-鿿]", behavior="isolated")
sp.pre_tokenize_str("これは日本語の文章です。")
# → [("これは日本語の文章です。", (0, 12))]    ← one pre-token, no matches

# Wrap in Regex():
from tokenizers import Regex
sp2 = Split(pattern=Regex("[぀-ヿ一-鿿]"), behavior="isolated")
sp2.pre_tokenize_str("これは日本語の文章です。")
# → [("こ", (0,1)), ("れ", (1,2)), ("は", (2,3)), ...]   ← works as intended
```

The `str` version finds zero matches and returns the input as one piece. The `Regex()`-wrapped version isolates each character correctly.

## Why HuggingFace `tokenizers` does this

It's not a library bug — it's an intentional API design. The `Split` API supports both literal-substring matching (useful when you actually want to split on a specific delimiter like `"|"` in a CSV) and regex matching (when you want pattern-based splitting). The choice is conveyed by the *type* of `pattern`:

- `pattern: str` → literal substring
- `pattern: Regex(str)` → regex pattern

The DX footgun is that the library accepts a `str` containing regex metacharacters without warning. There's no signal that the pattern looks like a regex; it's just silently treated as a literal substring.

## Prior reports

The same DX issue has been reported to the HuggingFace `tokenizers` repo multiple times:

- [#1369](https://github.com/huggingface/tokenizers/issues/1369): "BPE tokenization model does not respect custom RegEx via Split pre-tokenizer" — closed as "Not planned" (behavior change off the table)
- [#1264](https://github.com/huggingface/tokenizers/pull/1264): docs improvement (merged)
- [#1565](https://github.com/huggingface/tokenizers/issues/1565): docs improvement (merged)

We've added our experience to #1369 and filed a separate proposal:

- [#2109](https://github.com/huggingface/tokenizers/issues/2109) — proposal to emit a `UserWarning` from `pre_tokenize_str()` when a `str` pattern produces zero matches. Non-breaking debug-time surface that catches exactly this silent-failure mode.

## What the fix actually changes

After applying `Regex()` and retraining all 102 tokenizers:

### Per-script chars/token on FLEURS-test

| Script | Before | After | Change |
|---|---:|---:|---:|
| Brahmic / Abugida (17 langs) | 5.08 | 1.27 | **−75%** |
| Hebrew/Armenian/Ethiopic/Georgian (4) | 4.78 | 3.31 | −31% |
| CJK (4) | 2.14 | 1.57 | −27% |
| Cyrillic (10) | 5.57 | 4.45 | −20% |
| Arabic-script (5) | 5.11 | 4.21 | −18% |
| Latin (62) | 5.54 | 4.71 | −15% |
| **Overall median** | **5.35** | **4.44** | **−17%** |

The big wins are concentrated on non-Latin scripts. Latin scripts barely change because GPT-2-style word splitting partially worked even with the bug (via the `ByteLevel` fallback).

### Downstream impact on FFT

In a 5-language ablation (arabic + 4 CJK, seed=2024, paper-matched config) with *only* the tokenizer fix applied:

- **Japanese** combined CER: 50.49% → 40.22% (**−10 points**, rescues the documented FFT failure)
- **Japanese** combined WER: 106.94% → 91.87% (**−15 points**)

Other 4 langs in the ablation regressed slightly because of interactions between the tokenizer change and their existing training recipe.

## See also

- [Tokenizers → Recipe](recipe.md) for the full BPE training recipe
- [Tokenizers → Audit methodology](audit-methodology.md) for how we measure cross-boundary merges and chars/token
- [Findings → Ptfix 5-lang ablation](../findings/ptfix-5lang-ablation.md) for the downstream FFT impact
- [vamsin07/multilingual-bpe-tokenizers](https://github.com/vamsin07/multilingual-bpe-tokenizers) — public repo with the fix already applied
- HuggingFace tokenizers [#2109](https://github.com/huggingface/tokenizers/issues/2109) — upstream proposal
