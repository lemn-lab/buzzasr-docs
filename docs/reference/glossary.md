# Glossary

| Term | Meaning |
|---|---|
| **SFT** | Plain fine-tuning of Whisper on a language's speech, keeping Whisper's tokenizer. |
| **FFT** | Fine-tuning with a native tokenizer: train a new tokenizer for the language, warm-start its embeddings, then fine-tune with some text mixed in. |
| **Native tokenizer** | A tokenizer trained on the target language, replacing Whisper's English-heavy one. |
| **Text mixing** | Interleaving text-only examples with speech during training, so the new vocabulary is well grounded. |
| **Warm-start** | Initializing the new token embeddings from Whisper's existing ones instead of at random. |
| **Whisper zero-shot** | Whisper-large-v3 used as-is, with no fine-tuning. The baseline we compare against. |
| **CER / WER** | Character / word error rate. Lower is better. |
| **FLEURS** | A 102-language speech benchmark, used here for training and evaluation. |
| **CommonVoice** | Mozilla's multilingual speech corpus, used as additional training and evaluation data. |
| **goldfish** | The multilingual text corpus (Chang et al.) used to train the tokenizers and for text mixing. |
