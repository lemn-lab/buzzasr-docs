---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>
<p class="kicker">BuzzASR</p>
<h1>100+ monolingual speech recognizers, one per language</h1>
<p class="tag">We fine-tune Whisper-large-v3 into a separate model for each of 102 FLEURS languages, with two recipes: plain fine-tuning (SFT) and full fine-tuning with a native tokenizer (FFT). This site is a short technical overview of the project: the two recipes, why the tokenizer matters, and what the models achieve.</p>
</div>

<div class="stats">
  <div class="stat"><div class="n">89 / 102</div><div class="l">languages where the best BuzzASR model beats Whisper-large-v3 zero-shot (FLEURS-test, CER)</div></div>
  <div class="stat"><div class="n">31 / 102</div><div class="l">lowest CER among open systems on FLEURS-test</div></div>
  <div class="stat"><div class="n">41 / 102</div><div class="l">lowest CER on the combined FLEURS + CommonVoice set</div></div>
</div>

!!! note "How to read the SOTA counts"
    "Lowest CER among open systems" means the best of SFT or FFT (picked per language on validation, never on test) beats Omni-1B/7B, MMS-1B, Qwen3-ASR, and Cohere on CER. We report it on FLEURS-test (31) and on the combined FLEURS + CommonVoice set (41).

## The two recipes

FFT's advantage comes from its tokenizer. Whisper's tokenizer was built for English-heavy data, so it splits other languages into many small pieces. FFT trains a native tokenizer that keeps whole morphemes. That one change makes FFT more accurate, and it also makes decoding faster.

<figure class="demo" markdown>
<iframe class="anim" src="assets/decode_lanes.html" height="560" loading="lazy" title="SFT vs FFT decoding on Estonian"></iframe>
<figcaption>Decode latency is token count times time-per-token. Whisper-ZS and SFT share a tokenizer and emit the same number of tokens, so they tie. FFT's native tokenizer emits fewer, so it finishes first.</figcaption>
</figure>

<figure class="demo" markdown>
<iframe class="anim" src="assets/fft_pipeline.html" height="600" loading="lazy" title="FFT tokenizer replacement"></iframe>
<figcaption>How FFT swaps in a native tokenizer without retraining from scratch: learn a tokenizer, warm-start its embeddings from Whisper's, then fine-tune.</figcaption>
</figure>

## Where to start

<div class="grid cards" markdown>

-   __Technical overview__

    What BuzzASR is, and how the two recipes (SFT and FFT) differ.

    [Read the overview →](overview.md)

-   __The tokenizer__

    Why a native tokenizer helps, and a bug worth knowing about.

    [Tokenizer →](tokenizer.md)

-   __Results__

    How accurate the models are, and why FFT decodes faster.

    [See results →](results.md)

-   __Models__

    Per-language weights on Hugging Face, released as they are finalized.

    [Get the models →](models.md)

</div>

Model weights are being released on Hugging Face as the per-language models are finalized. See [Models](models.md).

## Acknowledgments

Developed in collaboration with [EleutherAI](https://www.eleuther.ai/), whose compute and support made these training runs possible.

<a class="collab" href="https://www.eleuther.ai/" target="_blank" rel="noopener" title="EleutherAI">
  <img src="assets/eleutherai.png" alt="EleutherAI logo" loading="lazy" />
</a>

## Citation

```bibtex
@misc{buzzasr,
  title  = {BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models},
  author = {Shivam Singh and Aditya Yadavalli and Catherine Arnett and Alex Warstadt},
  year   = {2026},
  note   = {Findings of the Association for Computational Linguistics: EMNLP 2026}
}
```
