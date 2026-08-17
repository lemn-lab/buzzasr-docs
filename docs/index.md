---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>
<p class="kicker">BuzzASR</p>
<h1>100+ monolingual speech recognizers, one per language</h1>
<p class="tag">We fine-tune Whisper-large-v3 into a separate specialist for each of 102 FLEURS languages, using two recipes: plain fine-tuning (SFT), and full fine-tuning with a native tokenizer (FFT). This site documents how the code actually works, not just what the paper says.</p>
</div>

<div class="stats">
  <div class="stat"><div class="n">88 / 102</div><div class="l">languages where the best BuzzASR model beats Whisper-large-v3 zero-shot (FLEURS-test, CER)</div></div>
  <div class="stat"><div class="n">32 / 102</div><div class="l">lowest CER among open systems on FLEURS-test</div></div>
  <div class="stat"><div class="n">41 / 102</div><div class="l">lowest CER on the combined FLEURS + CommonVoice set</div></div>
</div>

!!! note "How to read the SOTA counts"
    "Lowest CER among open systems" compares the best of SFT/FFT (chosen per language on validation, no test peeking) against Omni-1B/7B, MMS-1B, Qwen3-ASR and Cohere. FLEURS-test (32) is the clean apples-to-apples number. The combined count (41) is higher partly because the baselines are zero-shot on CommonVoice while BuzzASR is trained on it — a real gain, but an in-domain one worth stating.

## The two recipes, in motion

FFT's whole advantage is its tokenizer. Whisper's tokenizer was built for English-heavy data, so it splits other languages into many small pieces; FFT trains a native one that keeps whole morphemes. That single change is what makes FFT both more accurate and faster to decode.

<figure class="demo" markdown>
<iframe class="anim" src="assets/decode_lanes.html" height="560" loading="lazy" title="SFT vs FFT decoding on Estonian"></iframe>
<figcaption>Decode latency = token count × time-per-token. Whisper-ZS and SFT share a tokenizer and tie; FFT's native tokenizer decodes fewer tokens, so it finishes first.</figcaption>
</figure>

<figure class="demo" markdown>
<iframe class="anim" src="assets/fft_pipeline.html" height="600" loading="lazy" title="FFT tokenizer replacement"></iframe>
<figcaption>How FFT swaps in a native tokenizer without retraining from zero: learn a tokenizer, warm-start its embeddings from Whisper's, then fine-tune.</figcaption>
</figure>

## Where to start

<div class="grid cards" markdown>

-   __Reproduce a run__

    Set up the environment and train one FFT model end to end.

    [Getting started →](getting-started/single-fft-job.md)

-   __Understand the pipeline__

    How the dispatcher, per-job scripts, and training loop fit together.

    [Pipeline overview →](pipeline/overview.md)

-   __The tokenizer__

    How the per-language BPE tokenizers are built, and the Split bug that broke the first batch.

    [Tokenizers →](tokenizers/recipe.md)

-   __Decoder latency__

    Why FFT decodes faster, and what actually drives it.

    [Latency finding →](findings/decoder-latency.md)

</div>

## What this documents

The goal is that someone new to the project — or your future self — can reproduce results from scratch, follow the training pipeline end to end, change a recipe safely, and extend it to a new language. It describes the real code paths, the failure modes we hit, and the fixes.

## Built for one multi-GPU node

BuzzASR trains 100+ specialists on a single machine with several GPUs. The scheduler is a small claim-file dispatcher: a worker claims the next language by atomically renaming a file, so any number of workers share the GPUs without a queue server and without racing each other. If you have a multi-GPU box and a long list of jobs, the pattern is worth stealing.

[How the dispatcher works →](pipeline/dispatcher.md)

Model weights will be released on Hugging Face once the models are finalized.

## Citation

```bibtex
@misc{buzzasr,
  title  = {BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous submission, under review}
}
```
