# Per-job script generation

How `train_fft.py` turns the per-lang template into a runnable training script.

## The two-layer pattern

Each language has its own training template at:

```
_pod_share/jobs/train_strategy_c_<lang>.py
```

This is a *complete* training script with hard-coded constants for that language (FLEURS code, tokenizer path, goldfish corpus path, language-specific eval logic, etc.). It's runnable directly — but with default seed, default output dir, default LR multipliers, etc.

`train_fft.py` is an *orchestrator* that customizes the template for each run. It:

1. Reads `train_strategy_c_<lang>.py`
2. Text-substitutes ~10 constants
3. Writes the patched script to `run_state/fft_scripts/train_fft_<lang>_cfg<X>_s<N>_main.py`
4. `subprocess.call([python, that_script])` and waits

The pattern means the *actual training code* per language is a real Python file you can run by hand. No abstractions, no `if config["x"]:` branches inside a generic script. Just a flat per-lang program with whatever language-specific tweaks the recipe needs.

## What gets patched

```python
CONFIGS = {
    "A": {"embed_lr": 3.0, "dec_lr": 3.0},      # aggressive
    "B": {"embed_lr": 1.0, "dec_lr": 1.0},      # conservative
    "C": {"embed_lr": 3.0, "dec_lr": 1.0},      # embed-heavy
    "D": {"embed_lr": 2.0, "dec_lr": 2.0},      # middle (added for scaling matrix)
}

def patch_template(template_path, cli_args, out_path):
    s = template_path.read_text()
    cfg = CONFIGS[cli_args.config_id]
    
    # LR multipliers
    s = replace_line(s, r"^EMBED_LR_MULT\s*=.*$",
                     f"EMBED_LR_MULT = {cfg['embed_lr']}  # matrix cfg={cli_args.config_id}")
    s = replace_line(s, r"^DECODER_LR_MULT\s*=.*$",
                     f"DECODER_LR_MULT = {cfg['dec_lr']}  # matrix cfg={cli_args.config_id}")
    
    # ASR ratio + text-MTL
    s = replace_line(s, r"^ASR_RATIO\s*=.*$",
                     f"ASR_RATIO = {cli_args.asr_ratio}  # matrix-injected")
    s = replace_line(s, r"^GOLDFISH_PATH\s*=.*$",
                     f"GOLDFISH_PATH = \"{cli_args.text_mtl_path}\"")
    
    # Output dir
    s = replace_line(s, r"^OUTPUT_DIR\s*=.*$",
                     f"OUTPUT_DIR = Path(\"{cli_args.output_dir}\")")
    
    # Seed (injected at top of main())
    seed_block = f"    random.seed({cli_args.seed}); np.random.seed({cli_args.seed}); torch.manual_seed({cli_args.seed})\n"
    s = re.sub(r"(\ndef main\(\):\n)", r"\1" + seed_block, s, count=1)
    
    # Run-specific tmp dir (avoid collisions when multiple jobs share /tmp)
    s = re.sub(r"ext_dir = Path\(.+?\)",
               f"ext_dir = Path(\"/tmp/replacement-tokenizer-{cli_args.fleurs_code}_{cli_args.config_id}_{cli_args.seed}\")",
               s, count=1)
    
    out_path.write_text(s)
```

The patched script can then be `python`'d directly, including for debugging or rerunning after edits.

## Why text substitution and not Jinja / a config system

- **Debuggability**: the patched script is a real Python file. `python -i run_state/fft_scripts/...py` works for stepping through.
- **Diff visibility**: `diff template patched` shows exactly what changed. No hidden config-merge logic.
- **Language-specific tweaks survive**: per-lang templates can have any language-specific code (e.g., a special pre/post-processing step for Hebrew). A generic config-driven runner would force these into branches.
- **Trivial to add a new lang**: copy a template, edit the constants. No schema migration.

The cost is that template drift between languages is real — see [Recipes → Save criteria](../recipes/save-criteria.md) for an example of how the templates diverged into three save-criterion groups.

## Where the patched scripts live

```
run_state/fft_scripts/
├── train_fft_spanish_cfgB_s42_main.py
├── train_fft_spanish_cfgB_s1337_main.py
├── train_fft_japanese_cfgA_s42_main.py
├── train_fft_japanese_cfgA_s1337_main.py
└── ... (one per matrix entry)
```

These persist after training so you can re-run or inspect any past job. Combined with the matrix.json + status.json snapshots, this is the audit trail.

## See also

- [Training loop internals](training-loop.md) — what the patched script does once running
- [Eval + checkpoint mechanics](eval-checkpoint.md) — what gets saved
