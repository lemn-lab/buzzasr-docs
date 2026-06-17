# Audit methodology

How we measure tokenizer quality and detect failure modes.

## Compression rate: chars/token on FLEURS-test

```python
from tokenizers import Tokenizer

def chars_per_token(tokenizer_path, sentences):
    tok = Tokenizer.from_file(str(tokenizer_path))
    total_chars = total_tokens = 0
    for s in sentences:
        enc = tok.encode(s)
        total_chars += len(s)
        total_tokens += len(enc.ids)
    return total_chars / total_tokens
```

Higher is better (fewer tokens per char of text → shorter decoder sequences → faster inference).

Note: chars/token computed on the *training* corpus and chars/token computed on a *held-out* corpus (e.g. FLEURS-test) should be similar. A large divergence indicates the tokenizer has memorized training-specific multi-character tokens.

## Cross-boundary merge audit

Each BPE merge produces a token. In a properly pre-tokenized pipeline, no token should span multiple pre-token boundaries. A pre-token in a byte-level recipe has one of these structures:

- `Ġword` (Ġ + letters)
- `word` (letters only)
- `Ġnumbers` (Ġ + 1–3 digits)
- `Ġpunct` (Ġ + punctuation)
- etc.

A *cross-boundary* token violates this structure. For example:

- `ĠwordĠword` — contains two `Ġ`s, so it spans two word boundaries
- `wordpunct` — contains both letters and punctuation, with no `Ġ` between
- `word\nword` — contains an embedded newline

```python
def is_cross_boundary(token_str):
    # Multiple Ġs = spans multiple words
    if token_str.count('Ġ') > 1: return True
    # Embedded newline
    if any(c in token_str for c in '\r\n'): return True
    # Ġ at non-zero position
    if 'Ġ' in token_str and not token_str.startswith('Ġ'): return True
    # Mixed character classes in the body
    body = token_str.lstrip('Ġ')
    classes = {classify(c) for c in body}    # 'L' (letter), 'N' (digit), 'P' (punct)
    return len(classes) > 1
```

A healthy tokenizer should have <5% cross-boundary merges. A broken one (like our pre-fix tokenizers) can have 60–98%.

## Per-script grouping

When reporting medians, we group langs by script family because the impact of pre-tokenization is highly script-dependent:

- **CJK** (Mandarin, Cantonese, Japanese, Korean): char-isolation is critical; chars/token floor is ~1.0
- **Abugida / Brahmic** (Hindi, Bengali, Tamil, Khmer, Lao, …): char-isolation gives biggest wins
- **Arabic-script** (Arabic, Urdu, Persian, …): word-level pre-tokenization via `\p{L}+`; chars/token ~4 is healthy
- **Hebrew / Armenian / Georgian / Ethiopic**: mixed isolation + word-level
- **Cyrillic**: word-level; chars/token ~4–5
- **Latin**: word-level; chars/token ~5–6

## Verification chain (for the Regex() fix)

To verify the fix landed correctly:

1. Train tokenizer with `Split(pattern=Regex(SPLIT_PATTERN), ...)`
2. For each lang, compute chars/token on FLEURS-test
3. Compare to pre-fix numbers — should be substantially lower for non-Latin scripts
4. Sample tokens from FLEURS-test encoding and inspect: no multi-word phrases, CJK tokens are mostly 1 char each

For details on what the fix delivered, see [Split bug and fix](split-bug-and-fix.md).

## See also

- [Tokenizers → Recipe](recipe.md) — the training pipeline
- [Tokenizers → Split bug and fix](split-bug-and-fix.md) — the headline case study
