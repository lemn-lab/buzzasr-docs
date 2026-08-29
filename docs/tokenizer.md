# The tokenizer

FFT's advantage comes from replacing Whisper's tokenizer with one trained on the target language. This page explains why that helps, and a bug we ran into that is worth knowing about if you build tokenizers this way.

<figure class="demo" markdown>
<iframe class="anim" src="assets/fft_pipeline.html" height="600" loading="lazy" title="FFT tokenizer replacement"></iframe>
<figcaption>Learn a native tokenizer, warm-start its embeddings from Whisper's, then fine-tune.</figcaption>
</figure>

## Why a native tokenizer

A tokenizer splits text into the units a model reads and writes. Whisper's was built mostly on English, so for many other languages it uses short, generic pieces and needs a lot of them to spell a word. A tokenizer trained on the target language keeps larger, meaningful units, so the same sentence becomes far fewer tokens.

Fewer tokens per sentence helps in two ways. The model has an easier sequence to predict, which tends to improve accuracy on non-Latin scripts, and it has fewer tokens to generate at decode time, which makes it faster (see [Results](results.md)).

We keep the native tokenizer compatible with Whisper: the same vocabulary size and the same special tokens, so it drops in without changing the rest of the model.

## How much more compact

On held-out text, the native tokenizers are much more compact than Whisper's for non-Latin scripts, and about the same for Latin scripts, where Whisper was already reasonable:

| Script family | Compression vs Whisper |
|---|---|
| Brahmic / Abugida | ~75% fewer tokens |
| Hebrew / Armenian / Ethiopic / Georgian | ~30% fewer |
| CJK | ~25% fewer |
| Cyrillic | ~20% fewer |
| Arabic-script | ~18% fewer |
| Latin | ~15% fewer |

## A bug worth knowing about

Our first tokenizers had a silent bug. The library we use accepts a splitting rule either as a plain string or as a regular expression, and it treats a plain string as literal text to match. We passed our rule as a plain string, so it was never read as a pattern and effectively did nothing. Training still ran and produced a tokenizer, but not the one we intended: it merged text freely across word and character boundaries, and on small corpora it memorized whole phrases as single tokens.

The fix was one line, wrapping the rule so the library reads it as a regular expression:

```python
# before: the rule is treated as literal text, so it never matches
Split(pattern=RULE, behavior="isolated")

# after: the rule is treated as a regular expression
Split(pattern=Regex(RULE), behavior="isolated")
```

After the fix and a retrain, the tokenizers became much more compact on non-Latin scripts (the table above), and one language that had failed outright under the old tokenizer started working again. The general lesson: when a tokenizer behaves oddly, first check that your splitting rule is actually being applied.

## See also

- [Technical overview](overview.md): how the tokenizer fits into the FFT recipe.
- [Results](results.md): what the tokenizer change does to accuracy and speed.
- [multilingual-bpe-tokenizers](https://github.com/vamsin07/multilingual-bpe-tokenizers): the tokenizer recipe as a public repo.
