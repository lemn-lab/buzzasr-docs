# Decoder latency

FFT is not just more accurate than SFT and Whisper zero-shot — it is faster to decode, and the reason is entirely the tokenizer.

<figure class="demo" markdown>
<iframe class="anim" src="../../assets/decode_lanes.html" height="560" loading="lazy" title="SFT vs FFT decoding on Estonian"></iframe>
<figcaption>Whisper-ZS and SFT share a tokenizer and finish together; FFT's native tokenizer emits fewer tokens for the same sentence.</figcaption>
</figure>

## Latency is a token-count problem

Time per token (TPT) is fixed by the model and comes out the same, near 19.6 ms, whether we run Whisper zero-shot, SFT, or FFT (we measured it separately for each). So decode latency is:

```
latency ≈ fixed overhead (~50 ms) + TPT × (tokens generated)
```

With TPT held constant, the only thing that moves latency is how many tokens a system emits. Whisper-ZS and SFT keep Whisper's tokenizer and emit the same number, so they tie. FFT replaces the vocabulary with one trained on the target language and emits far fewer.

## Per-script speed-up

Median per-utterance latency on the combined FLEURS + CommonVoice test set, versus Whisper zero-shot:

| script | n | zs (ms) | sft (ms) | fft (ms) | fft vs zs |
|---|---|---|---|---|---|
| Latin | 61 | 714 | 714 | **448** | 1.59× |
| Cyrillic | 10 | 785 | 788 | **449** | 1.75× |
| Arabic | 6 | 964 | 1101 | **487** | 1.98× |
| Brahmic | 16 | 1611 | 2771 | **1384** | 1.16× |
| Sino-Japanese | 4 | 502 | 541 | **514** | 0.98× |
| Other | 5 | 1475 | 1861 | **420** | 3.51× |
| **Global** | 102 | 785 | 818 | **503** | **~1.6×** |

FFT is faster on 96 of 102 languages. SFT gives essentially no speed-up (0.96× global) and is *slower* than zero-shot on Brahmic — same tokenizer, and fine-tuning makes it transcribe the full utterance where zero-shot bails early. **The latency win belongs to the vocabulary replacement, not to fine-tuning.**

## Static compression is not inference compression

It is tempting to read the speed-up off the tokenizer's compression ratio on text. That over-predicts it, because two different things share the tokenizer:

- **Static compression** — tokenize the ground-truth transcript. A property of the tokenizer and the text; the model is not involved. For Brahmic scripts the native tokenizer is about **7.9× more compact** than Whisper's.
- **Inference compression** — how many tokens the model actually emits while transcribing the audio. A property of the model's prediction, usually different.

Latency tracks the second, not the first:

| predictor | correlation with FFT speed-up |
|---|---|
| static (tokenizer) compression | +0.15 |
| inference (generated-token) compression | **+1.00** |

Brahmic makes the gap concrete: the tokenizer is 7.9× more compact, but latency improves only ~1.4×, because the model — trained on limited paired audio — never emits those long whole-word tokens, and zero-shot truncates the output entirely. The compact vocabulary exists; it just doesn't get used at inference.

## Reliability

The native tokenizer also almost eliminates catastrophic repetition loops:

| system | loop failures (FLEURS-test) |
|---|---|
| Whisper zero-shot | 84 |
| SFT | 68 |
| **FFT** | **2** |

## Reproduce

`bench_latency_v4.py` (FLEURS with `--tag clean`, CommonVoice with `--cv --tag cv`), then `latency_combined_report.py`; the static-vs-inference correlation is in `latency_drivers.py`. Decoding matches the accuracy eval: greedy, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`.
