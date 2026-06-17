# Training loop internals

What actually happens between "job claimed by GPU 0" and "Step 100: WER=29.59%" appearing in the log.

## CPU-bound init phase (the "0 MiB GPU" minutes)

When a job hits step 0 but `nvidia-smi` still shows 0 MiB used on its GPU, this is normally 3–5 minutes of setup:

### 1. Tokenizer build (Strategy C)

Load Whisper's tokenizer + the per-language BPE tokenizer (from `frankenstein_fix/`). Compute the overlap of token *strings* between them. For each token in the new vocab:

- If it's in both vocabs → port the existing embedding row
- If it's new (lang-only) → initialize from a multivariate normal sampled from the empirical mean/covariance of Whisper's embeddings

This warm-start init is better than random init because new tokens "land" in the right region of embedding space.

**~30 seconds.**

### 2. Model load

```python
WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
```

Reads safetensors from `/mnt/ssd-3/hf_cache/` into CPU RAM. ~6 GB FP16 weights → ~6 seconds at NFS read speed.

### 3. Embedding resize

This is the FFT-specific step. Whisper's embedding matrix is `(51866, 1280)`. Per-lang BPE has its own vocab plus Whisper's special tokens — final size `(53474, 1280)` in our recipe. The old embeddings are kept; new rows get initialized from the warm-start procedure above.

**~10 seconds.**

### 4. ASR dataset load

```python
asr_iid = load_from_disk(f"/mnt/ssd-3/asr/training_sets/{fleurs_code}")
```

That's the pre-built FLEURS+CV IID-shuffled Arrow dataset. For a typical 35k-sample lang (~2 GB of mel features when materialized), this takes 1–2 minutes.

### 5. Goldfish text-MTL load

If `ASR_RATIO < 1.0`, load up to `max_text_lines` from the per-lang goldfish file:

```python
text_train = GoldfishTextDataset(
    text_mtl_path,
    tokenizer,
    max_lines=500_000,
)
```

For Mandarin/Japanese/Arabic (6–10 GB goldfish files), the first 500k lines is most of the cost — file-system reads are the bottleneck, not the tokenization itself.

**1–3 minutes.**

### 6. Model → GPU

```python
model.to("cuda:0")
```

The actual CPU→GPU transfer over PCIe. Whisper-large-v3 in FP32 is ~6 GB; the transfer takes ~3–6 seconds at PCIe Gen4 speeds (~16 GB/s real-world throughput).

### 7. Gradient checkpointing

```python
model.gradient_checkpointing_enable()
```

Trades compute for memory by *not* storing intermediate activations during forward, recomputing them on backward. Cuts memory ~40%, slows training ~25%. Whisper-large-v3 fine-tuning is memory-bound enough that we always enable it.

!!! tip "Why the 'staggered init' pattern"
    When you see 4 of 8 GPUs hot at 38 GB and the other 4 at 0 MiB, it's because the 4 slow ones are still loading their goldfish text corpus. Mandarin/Japanese/Arabic init is slowest because of corpus size. They'll catch up in a minute.

## Steady-state training step

Each "Step N" in the log is **one optimizer step**, but underneath it's `N × grad_accum` forward+backward passes. For cfg B (`grad_accum=24, batch_size=2`) that's 48 audio samples flowing through per optimizer step.

### The mixed-precision dance per micro-batch

```python
with autocast():                                    # (1) FP16 forward
    out = model(input_features, labels=labels)
    loss = out.loss

scaler.scale(loss / grad_accum).backward()           # (2) scaled backward
                                                     # gradients accumulate in FP32 master

# (after grad_accum micro-batches)
scaler.unscale_(optimizer)                           # (3) unscale before clipping
clip_grad_norm_(model.parameters(), 1.0)             # (4) gradient clipping
scaler.step(optimizer)                               # (5) skipped if any inf/nan
scaler.update()                                      # (6) adjust scale factor
scheduler.step()                                     # (7) cosine LR decay
optimizer.zero_grad()
global_step += 1
```

1. **`autocast()`**: FP16 forward (saves memory + faster on Ampere/Hopper tensor cores)
2. **`scaler.scale(loss).backward()`**: prevents FP16 grad underflow by scaling the loss up before backward; gradients are accumulated in FP32 master weights
3. **`scaler.unscale_(optimizer)`**: removes the scaling before clipping (since `clip_grad_norm_` expects unscaled grads)
4. **`clip_grad_norm_(params, 1.0)`**: standard global L2 clip — prevents single-batch gradient spikes from blowing up training
5. **`scaler.step(optimizer)`**: applies grads through the optimizer (AdamW). **If any grad is inf/nan, this is silently skipped** — this is the FP16 numeric stability mechanism
6. **`scaler.update()`**: adjusts the loss-scale factor based on whether the last few steps had inf/nan
7. **`scheduler.step()`**: advances the LR schedule

### Why GPU utilization swings

GPU utilization swings between 30–70% in `nvidia-smi` partly because of:

- **DataLoader bottleneck**: forward passes are GPU-bound, but the DataLoader (running in 2 worker processes) does CPU-side audio decoding + tokenization between batches. If the workers can't keep up (especially when text-MTL is mixed in — the text branch fetches a *different* batch type), the GPU briefly idles. That's the 0% util periods you see.
- **Eval pauses**: during eval (every 100 steps), the loop pauses for greedy generation; GPU util drops to ~0% while jiwer computes WER on CPU.
- **Optimizer step + scheduler step**: brief CPU activity between micro-batches.

There's headroom if anyone wanted to optimize this — NVIDIA DALI for audio preprocessing, TFRecord-style DataLoader, or just bumping `num_workers` from 2 to 4. None of those are blockers for current research.

## What `nvidia-smi` is telling you

For a healthy steady-state training job:

```
GPU memory: 38 GB used
GPU util:   varies 30–70%
```

- **38 GB used** = model + optimizer state + gradients. AdamW's `m` and `v` buffers double the model size in FP32 even when training in FP16, plus activations from gradient checkpointing — total works out to ~38 GB for Whisper-large-v3 at `batch_size=2, grad_accum=16`.
- **0% util** at a single snapshot = either between micro-batches (CPU DataLoader is the bottleneck for that step), during eval (greedy decode is intermittent), or during a WER compute (jiwer is CPU-bound).
- **40–70% util** during steady-state. We never see 95%+ because of gradient accumulation + DataLoader inefficiency.

## Where to look when things go wrong

| Symptom | Likely cause | Where to look |
|---|---|---|
| Stuck at "Original embedding norms" for >10 min | Loading goldfish corpus too slow | Check NFS health; goldfish file size |
| GPU mem 0 MiB, CPU% high, no progress | CPU init phase (normal) | Wait a few more min |
| `CUDA out of memory` | Effective batch too large for vocab size | Reduce `batch_size`, increase `grad_accum` |
| WER >100% in eval, model emits repeats | Rep-trap collapse | See [Recipes → Save criteria](../recipes/save-criteria.md) |
| inf/nan in loss every step | LR too high for FP16 | Lower `EMBED_LR_MULT` or use cfg B |
| Job exits at step 0 with TypeError | Tokenizer merges format mismatch | See [Tokenizers → Audit methodology](../tokenizers/audit-methodology.md) |
