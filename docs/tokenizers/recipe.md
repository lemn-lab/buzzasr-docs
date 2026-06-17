# Tokenizer recipe

The per-language byte-level BPE training recipe used for the FFT tokenizer-replacement strategy.

## Goal

Build a per-language BPE tokenizer that can drop in as a Whisper tokenizer replacement. This means:

- Same vocab size as Whisper (51,865 tokens) so the embedding matrix resize is straightforward
- Same special tokens (`<|startoftranscript|>`, `<|notimestamps|>`, 99 language tokens, timestamp tokens)
- Same byte-level encoding (Whisper's `Ġ`-as-space convention)
- Substantial compression gain on monolingual text

## The pipeline

```python
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, trainers, processors, decoders

tok = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),  # (*)
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.post_processor = processors.ByteLevel(trim_offsets=False)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=51865,
    min_frequency=5,
    max_token_length=16,
    special_tokens=SPECIAL_TOKENS,
)
tok.train_from_iterator(corpus_iter(goldfish_file, max_lines=200_000), trainer=trainer)
```

`(*) Regex()` is essential — see [Split bug and fix](split-bug-and-fix.md) for the long story.

### `SPLIT_PATTERN`

```python
CHAR_LEVEL_RANGES = (
    r"぀-ゟ"   # Hiragana
    r"゠-ヿ"   # Katakana
    r"一-鿿"   # CJK Unified Ideographs
    r"㐀-䶿"   # CJK Ext A
    r"가-힯"   # Hangul Syllables
    r"ሀ-፿"   # Ethiopic
    r"ऀ-ॿ"   # Devanagari
    r"฀-๿"   # Thai
    r"຀-໿"   # Lao
    r"Ⴀ-ჿ"   # Georgian
    r"஀-௿"   # Tamil
    r"ঀ-৿"   # Bengali / Assamese
    r"઀-૿"   # Gujarati
    r"਀-੿"   # Gurmukhi (Punjabi)
    r"ಀ-೿"   # Kannada
    r"ఀ-౿"   # Telugu
    r"ഀ-ൿ"   # Malayalam
    r"က-႟"   # Myanmar (Burmese)
    r"ក-៿"   # Khmer
    r"԰-֏"   # Armenian
    r"֐-׿"   # Hebrew
)
SPLIT_PATTERN = (
    f"[{CHAR_LEVEL_RANGES}]"
    r"|(?:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+"
)
```

The character-class block isolates each CJK / abugida / Hebrew / Armenian char as its own pre-token (so BPE can't merge across script characters). The alternation patterns after handle GPT-2-style word splitting for Latin/Cyrillic scripts.

### `min_frequency=5` and `max_token_length=16`

These two constraints prevent the BPE trainer from collapsing small/repetitive corpora into degenerate mega-tokens. Specifically:

- `min_frequency=5` — a merge candidate must appear at least 5 times before being applied. Discards rare merges that would only memorize specific training fragments.
- `max_token_length=16` — caps the length of any token at 16 chars. Hard ceiling against multi-word phrase memorization.

These were added after diagnosing the Kamba 18× compression cliff (see [Split bug and fix](split-bug-and-fix.md)). Kamba's full pre-training corpus is only ~3340 lines — small enough that BPE without these constraints memorizes whole sentences.

## Training data: per-language goldfish corpora

```python
GOLDFISH_DIR = Path("/mnt/ssd-3/asr/text_pretraining/goldfish")
# One file per FLEURS lang, e.g.:
#   spa_latn.txt   (Spanish)
#   arb_arab.txt   (Arabic)
#   yue_hant.txt   (Cantonese)
#   kam_latn.txt   (Kamba)
```

We use up to 200,000 lines per language (training plateau in our sweep). For corpora smaller than 200k lines, we use everything.

## Outputs

Each tokenizer is saved to `/mnt/ssd-3/checkpoints/frankenstein_fix/<lang>/tokenizer/`:

```
arabic/tokenizer/
├── tokenizer.json    # full HF tokenizer config
├── vocab.json        # token ↔ id map (Whisper convention)
└── merges.txt        # ordered BPE merges
```

## Time

- 102 langs in parallel (8 CPU workers): ~10 min wall-clock total
- Single lang on 1 CPU: ~30–90 seconds depending on corpus size

## See also

- [Split bug and fix](split-bug-and-fix.md) for the diagnosis of the API issue that affected the original tokenizers
- [Audit methodology](audit-methodology.md) for how we measure compression and cross-boundary merges
- [vamsin07/multilingual-bpe-tokenizers](https://github.com/vamsin07/multilingual-bpe-tokenizers) — public repo of this recipe
