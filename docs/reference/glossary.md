# Glossary

| Term | Meaning |
|---|---|
| **SFT** | Simple fine-tuning — Whisper fine-tune on monolingual ASR data with no vocab swap |
| **FFT** | Full fine-tuning — 3-stage pipeline with tokenizer replacement + multitask fine-tuning |
| **VFT** | Vanilla fine-tuning — internal naming for what's called SFT in the paper |
| **VFTMTL** | Vanilla fine-tuning + multi-task learning — SFT with text-MTL added |
| **MTL** | Multitask learning — interleave ASR examples with text-only examples |
| **Strategy C** | The FFT tokenizer-replacement + warm-start embedding init strategy |
| **Whisper-zs** | Whisper-large-v3 zero-shot (the baseline) |
| **cfg A/B/C/D** | LR-multiplier configurations for FFT — see [Hyperparameter selection](../recipes/hyperparameters.md) |
| **ASR_RATIO (α)** | Fraction of batches that are ASR vs. text-MTL. Default α=0.5 |
| **Rep-trap** | A training failure mode where loss keeps dropping but WER explodes due to autoregressive token repetition |
| **Combined-gate** | Save criterion that fires iff (WER improves OR loss improves) AND neither has drifted too far |
| **Group A / B / C** | The three per-lang save-criterion buckets in the deployed code |
| **frankenstein/** | Original per-lang BPE tokenizers (broken Split — pre-Regex fix) |
| **frankenstein_fix/** | Per-lang BPE tokenizers after the Regex() fix |
| **goldfish** | Multilingual text-pretraining corpus from Chang et al. used for text-MTL |
| **FLEURS** | Few-shot Learning Evaluation of Universal Representations of Speech — 102-language test bench |
| **CV / CV25** | Mozilla CommonVoice v25, used for additional ASR training data |
| **matrix.json** | Experiment manifest (list of all jobs to run) |
| **matrix_status.json** | Per-job runtime state (queued, running, done, failed) |
| **IID dataset** | Pre-combined FLEURS+CV training data, shuffled IID |
| **tokfix** | The 102-lang retrain using the fixed tokenizers (job_id suffix `_tokfix`) |
