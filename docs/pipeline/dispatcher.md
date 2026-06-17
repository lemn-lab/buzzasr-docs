# Dispatcher + matrix

The orchestration layer for running many fine-tunes in parallel across multiple GPUs and pods.

## `matrix.json` — the experiment manifest

A single JSON file listing every fine-tune job we want to run. One entry per (lang × cfg × seed × method) combination. Schema:

```python
{
    "job_id":         "fft_spanish_cfgB_s42_main",
    "method":         "fft",                    # or "vft", "vftmtl", "fft_strat_a", ...
    "lang":           "spanish",
    "fleurs_code":    "es_419",
    "whisper_lang":   "spanish",
    "config_id":      "B",                      # cfg A/B/C/D — see Recipes
    "seed":           42,
    "asr_ratio":      0.5,
    "max_text_lines": 500000,
    "text_mtl_path":  "/path/to/goldfish/spa_latn.txt",
    "output_dir":     "/checkpoints/matrix_runs/fft_spanish_cfgB_s42_main",
    "run_kind":       "main",                   # or "textmix_sweep", "scaling", ...
}
```

Across the project this grew to ~1200 entries (main FFT + textmix sweep + replication runs).

## `matrix_status.json` — the running state

A *separate* JSON keyed by `job_id`:

```python
{
    "fft_spanish_cfgB_s42_main": {
        "state":       "done",                  # or "queued", "running", "failed", "abandoned"
        "pod":         "shivam-0_tokfix",
        "pid":         3328333,
        "gpu":         2,
        "started_at":  "2026-06-17T22:48:00Z",
        "finished_at": "2026-06-17T26:12:00Z"
    },
    ...
}
```

The dispatcher only touches *its own* claimed entries — but reads the whole file every poll to know what's queued.

## How claims work

Both pods can run a dispatcher concurrently. Coordination is via `fcntl.flock`:

```python
import fcntl, json

def claim_job():
    with open("/mnt/ssd-3/asr/_repro_sync/run_state/claims.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)            # exclusive lock; blocks until acquired
        status = json.load(open("matrix_status.json"))
        # Find a "queued" entry, atomically flip it to "running"
        for job_id, st in status.items():
            if st["state"] == "queued":
                st["state"] = "running"
                st["pod"] = pod_name
                st["pid"] = os.getpid()
                json.dump(status, open("matrix_status.json", "w"))
                return job_id
        return None
    # lock auto-released on context exit
```

This works because the file is on the shared NFS mount and both pods see it. There's no race condition: only one pod can hold the lock at a time.

## Dispatcher invocation

```bash
python dispatcher.py \
    --pod shivam-0_tokfix \
    --gpus 0,1,2,3,4,5,6,7 \
    --methods fft \
    --matrix matrix_tokfix_only.json \
    --poll-interval 30
```

Args:

| Arg | Meaning |
|---|---|
| `--pod` | Pod identifier (just a label for `started_by` tracking) |
| `--gpus` | Which GPU IDs to use (claims one job per GPU) |
| `--methods` | Filter on `job["method"]` — only claim jobs of these types |
| `--matrix` | Filtered subset of `matrix.json` (useful for restricting to a specific run kind) |
| `--poll-interval` | Seconds between polls (default 30) |

## Per-job process tree

When the dispatcher claims a job and spawns it:

```
dispatcher.py (parent)
└── bash -c "CUDA_VISIBLE_DEVICES=2 python train_fft.py --lang spanish ..."
    └── train_fft.py (patches the template)
        └── python run_state/fft_scripts/train_fft_spanish_cfgB_s42_main.py (the real training)
```

The dispatcher records the **bash child PID** in `matrix_status.json` so the job can be killed from outside (via `kill -TERM <pid>`). The actual training subprocess (the leaf node) is not directly tracked — but killing the bash parent propagates SIGTERM down the tree.

## What happens on failure

If the leaf training process exits non-zero:

1. `train_fft.py` propagates the exit code up
2. `dispatcher.py` sees the bash child exited with nonzero
3. `matrix_status.json` flipped to `state="failed"`
4. Dispatcher moves on to next queued job

There is no automatic retry. Re-queueing is manual:

```python
import json
s = json.load(open("matrix_status.json"))
for job_id in [...]:
    s[job_id]["state"] = "queued"
json.dump(s, open("matrix_status.json", "w"))
```

## Why no real scheduler

We considered Celery, Slurm, Ray. For ~1200 jobs on 16 GPUs across 2 pods with shared NFS, the file-locked dispatcher is genuinely simpler:

- No external dependency
- No broker process to crash
- Status JSON is human-readable and editable
- Failure recovery is "edit the file"

The cost is no automatic retries, no priority queues, no fancy scheduling. Acceptable for a research workflow where you re-queue failures by hand after diagnosing them.

## See also

- [Per-job script generation](per-job-script.md) — what `train_fft.py` does next
- [Training loop internals](training-loop.md) — what the leaf process does
