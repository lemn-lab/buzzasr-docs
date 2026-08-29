# Models

We are releasing the trained models on Hugging Face as they are finalized, one model per language under the BuzzASR organization. This page links to each of them.

!!! info "Release in progress"
    Weights are being uploaded as the final per-language models are ready. This page will fill in as they land, so check back for the full set.

## Where to find them

All models live under the BuzzASR organization on Hugging Face, grouped in a single collection.

> **Collection:** _link coming with the first release_

Each language has its own model, named by language and recipe, so you can pull just the one you need.

## Using a model

Once a language is released, you can load it with the standard Whisper interface:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

name = "buzzasr/<model-name>"          # e.g. the Finnish model
model = WhisperForConditionalGeneration.from_pretrained(name)
processor = WhisperProcessor.from_pretrained(name)
```

## Available models

| Language | Recipe | Link |
|---|---|---|
| _released models will be listed here_ | | |
