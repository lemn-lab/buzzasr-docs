# Technical overview

BuzzASR is a set of monolingual speech recognizers, one per language. Rather than lean on a single multilingual model, we fine-tune Whisper-large-v3 separately for each of 102 FLEURS languages, so every model specializes in one language.

We use two recipes.

## SFT: plain fine-tuning

Standard fine-tuning of Whisper on speech for the target language. The model keeps Whisper's original tokenizer and vocabulary, and only its weights adapt. SFT is a strong baseline, especially for high-resource Latin-script languages where Whisper is already close to the best available systems.

## FFT: fine-tuning with a native tokenizer

Whisper's tokenizer was trained mostly on English text, so for many other languages it breaks words into a lot of small pieces. That makes the model read and write more tokens per sentence, which costs both accuracy and decoding speed.

FFT addresses this in three steps:

1. Train a new tokenizer on text for the target language, so it represents that language compactly.
2. Warm-start the new token embeddings from Whisper's, instead of starting them at random, so training begins from a sensible point.
3. Fine-tune on speech, mixing in some text-only examples so the new vocabulary is well grounded.

The result is a model that transcribes the language in fewer tokens. For most non-Latin scripts this improves accuracy, and it makes decoding noticeably faster.

## Choosing between them

For each language we pick the better of the two recipes on a validation set, with no peeking at the test set. As a rule of thumb, SFT tends to win where Whisper's tokenizer was already adequate (many Latin-script languages), and FFT tends to win where the tokenizer was the bottleneck (most other scripts).

## Training and evaluation data

Each model is trained on public speech for its language: FLEURS, plus CommonVoice where it is available. The tokenizer and the text-mixing step use monolingual text. We evaluate on held-out FLEURS and CommonVoice test sets.

## See also

- [The tokenizer](tokenizer.md): why the native tokenizer helps, and a bug worth knowing about.
- [Results](results.md): accuracy and decoding-speed highlights.
