# Results

Two headline results: BuzzASR is accurate relative to other open systems, and the FFT models decode faster because of the native tokenizer.

## Accuracy

We measure character error rate (CER), where lower is better, and pick the better of SFT and FFT per language (chosen on a validation set, never on test):

- On **89 of 102** languages, the best BuzzASR model beats Whisper-large-v3 zero-shot on FLEURS-test.
- On **31 of 102**, it has the lowest CER of any open system we compared against (Omni-1B/7B, MMS-1B, Qwen3-ASR, Cohere) on FLEURS-test.
- On **41 of 102**, it has the lowest CER on the combined FLEURS + CommonVoice set.

## Decoding speed

<figure class="demo" markdown>
<iframe class="anim" src="../assets/decode_lanes.html" height="560" loading="lazy" title="SFT vs FFT decoding"></iframe>
<figcaption>Whisper zero-shot and SFT share a tokenizer and emit the same number of tokens, so they finish together. FFT emits fewer tokens for the same sentence, so it finishes first.</figcaption>
</figure>

The time to produce one token is about the same across systems, so decoding time comes down to how many tokens a system emits. Because FFT's native tokenizer represents the language in fewer tokens, FFT decodes faster: roughly 1.6× faster than Whisper zero-shot on average, and faster on 96 of 102 languages. SFT, which keeps Whisper's tokenizer, gives essentially no speed-up.

One nuance: the speed-up is smaller than the tokenizer's raw compression would suggest. The tokenizer can represent text very compactly, but the model only benefits when it actually emits those compact tokens while transcribing, and for some low-resource scripts it does not fully use them.

## Reliability

The native tokenizer also almost eliminates a failure mode where the model gets stuck repeating itself. Across FLEURS-test, catastrophic repetition loops dropped from 84 cases under Whisper zero-shot and 68 under SFT to just 2 under FFT.

## See also

- [Technical overview](overview.md): the two recipes.
- [The tokenizer](tokenizer.md): why FFT emits fewer tokens.
