# Eval + checkpoint mechanics

What happens every `eval_steps` (typically 100) optimizer steps, and how a "best" checkpoint is chosen.

## The eval block

```python
if global_step % eval_steps == 0:
    metrics = evaluate(model, tokenizer, eval_loader, device, whisper_lang)
    # metrics: {"wer", "cer", "loss", "n"}
    
    # 1. Apply per-group save criterion (see "Save criteria" below)
    if save_criterion_met(metrics):
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": global_step,
            "wer": wer, "cer": cer, "eval_loss": eval_loss,
            "config_id": config_id, "seed": seed,
        }, BEST_MODEL_DIR / "checkpoint.pt")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 2. Patience-based early stop
    if patience_counter >= patience:  # typically 5 for FFT
        logger.info(f"Early stopping at step {global_step}")
        break
```

### What `evaluate()` does

```python
@torch.no_grad()
def evaluate(model, tokenizer, eval_loader, device, language):
    model.eval()
    refs, hyps, losses = [], [], []
    for batch in eval_loader:
        feats = batch["input_features"].to(device, dtype=torch.float16)
        labels = batch["labels"].to(device)
        with autocast():
            # Loss: teacher-forced (same as training)
            out = model(input_features=feats, labels=labels)
            losses.append(out.loss.item())
            # Greedy generation: autoregressive, one token at a time
            gen = model.generate(
                feats,
                max_new_tokens=256,
                language=language,
                task="transcribe",
                num_beams=1,
            )
        for g, t in zip(gen, batch["texts"]):
            hyps.append(tokenizer.decode(g, skip_special_tokens=True).strip())
            refs.append(t)
    model.train()
    return {
        "wer": compute_wer(refs, hyps),
        "cer": compute_cer(refs, hyps),
        "loss": float(np.mean(losses)),
        "n": len(refs),
    }
```

Two cost drivers:

1. **Greedy generation**: each utterance generates one token at a time, autoregressively. Encoder runs once (fixed ~20 ms for 30 s audio), then the decoder generates tokens until EOS or `max_new_tokens=256`. **This is where the tokenizer matters for in-training eval latency** — Whisper generates `<lang> <transcribe> <notimestamps> token_1 token_2 …`. For Brahmic langs with the old broken BPE, the model had to emit ~5× more tokens than with the fixed tokenizer to express the same transcription. So even in-training evals were artificially slow before the [Split bug fix](../tokenizers/split-bug-and-fix.md).

2. **WER/CER computation via `jiwer`**: CPU-bound, runs after all decoded. Typically 1–5 seconds for a ~500-sample val set.

For a typical lang with 280–980 samples, eval takes 3–8 minutes wall-clock.

## Save criteria

The save block applies one of three criteria depending on the language group:

### Group A: save-by-WER + `MAX_STEPS = 3000`

```python
if wer < best_wer:
    best_wer = wer
    best_loss = eval_loss
    best_cer = cer
    patience_counter = 0
    torch.save(...)
```

**17 langs** (the rep-trap recipe set):

- arabic, catalan, italian, polish, russian, german, spanish
- tamil, latvian, uzbek, cantonese, pashto, swahili, tajik
- georgian, ukrainian, galician

These are the langs where the rep-trap fix was empirically validated. The `MAX_STEPS=3000` cap means training stops at step 3000 regardless of patience — a hard backstop against catastrophic late-training collapse.

### Group B: combined-gate save + `MAX_STEPS = None`

```python
WER_TOL  = 0.05   # WER may drift up to 5 pp above best
LOSS_EPS = 0.05   # loss may drift up to 0.05 above best
improved_any   = (wer < best_wer) or (eval_loss < best_loss)
neither_diverged = (wer < best_wer + WER_TOL) and (eval_loss < best_loss + LOSS_EPS)
if improved_any and neither_diverged:
    best_loss = min(best_loss, eval_loss)
    best_wer  = min(best_wer, wer)
    best_cer  = min(best_cer, metrics["cer"])
    patience_counter = 0
    torch.save(...)
```

**82 langs** (the majority — everything not in Group A or C).

The combined gate saves whenever *either* metric improves, but blocks the save if *either* has drifted too far above its previous best. This catches the rep-trap collapse pattern where loss keeps dropping while WER explodes (and vice versa).

!!! info "Why combined-gate not just save-by-WER?"
    On the 82 lower-coverage langs, FLEURS-val is small (often <300 samples) so WER is noisy. Pure save-by-WER would either jitter randomly between saves or get stuck on a lucky early ckpt. The loss signal is much smoother. Combining them takes advantage of both.

### Group C: save-by-LOSS + `MAX_STEPS = None`

```python
if eval_loss < best_loss:
    best_loss = eval_loss
    best_wer = wer
    best_cer = metrics["cer"]
    patience_counter = 0
    torch.save(...)
```

**4 langs**: burmese, khmer, thai, ganda.

These are langs where WER is too unreliable to drive selection (script-specific issues with the WER metric: Thai has no whitespace word boundaries; Khmer and Burmese are abugida scripts where WER doesn't reflect transcription quality well). Loss-based selection is more reliable.

## Patience-based early stopping

When the save criterion *isn't* met, `patience_counter` increments. After `patience` consecutive unsuccessful evals, training stops.

- For FFT: patience = 5
- For SFT: patience = 3

At eval every 100 steps, that's at most 500 steps (FFT) or 300 steps (SFT) of "wasted" training before early-stop fires.

## Checkpoints on disk

### `best/checkpoint.pt`

The output that everything downstream cares about. Single file, ~6 GB. Format:

```python
{
    "model_state_dict": ...,    # full FP32 weights
    "step": ...,                # training step at save time
    "wer": ...,                 # the val WER that triggered the save
    "cer": ...,
    "eval_loss": ...,
    "config_id": ...,           # which cfg this ckpt belongs to
    "seed": ...,
}
```

Overwritten each new best. The atomic-write story isn't perfect — `torch.save` writes to the target path in chunks, so a SIGKILL mid-save can leave a truncated file. Empirically this hasn't bitten us, but it's a latent risk.

### `latest/checkpoint.pt`

Periodic checkpoint saved every `checkpoint_every` steps (typically 1000). Used for preemption recovery (restart from `latest/` if the job dies). On our infra we don't have preemption (no spot instances), so this is mostly dead weight; could be turned off to save ~6 GB per run.

### Loading a checkpoint downstream

```python
ckpt = torch.load("path/to/best/checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
```

`strict=False` is used because the *receiving* model architecture might have a different embedding size than the *saved* model (e.g., when loading an old-tokenizer ckpt into a new-tokenizer architecture). For valid same-architecture reloads, the load is exact.

## Disk-management notes

After a training run finishes:

- `best/checkpoint.pt` is the only required output
- `latest/checkpoint.pt` can be deleted (dead weight)
- The log file at `logs/matrix/<job_id>.log` is the audit trail — keep it

Storage budget per ckpt:

- Whisper-large-v3 FP32 state dict: ~6 GB
- 102 langs × 2-3 ckpts each (best + sometimes a fallback) → ~1.5 TB
- The full matrix (all variants) easily hits 5+ TB

## See also

- [Recipes → Save criteria](../recipes/save-criteria.md) for the methodology behind A/B/C groups
