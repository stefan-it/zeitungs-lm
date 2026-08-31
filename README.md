# Zeitungs-LM

![Zeitungs-LM](zeitungs-lm-v1.png)

This repository hosts all necessary information for benchmarking the [Zeitungs-LM](https://huggingface.co/stefan-it/zeitungs-lm-v1), a language model trained on Historical German Newspapers.

Technically the model itself is an ELECTRA model, which was pretrained with the [TEAMS](https://aclanthology.org/2021.findings-acl.219/) approach.

## Setup

This repo uses [uv](https://docs.astral.sh/uv/) for dependency management. The default
install pulls a CPU build of PyTorch:

```bash
uv sync
```

For training models on GPU, use this;

```bash
uv sync --no-group cpu --extra cuda
```

The GPU wheels are Linux-only; on other platforms these extras fall back to the
CPU build, so local development still works. Run any script with `uv run`, e.g.
`uv run python benchmark/script.py`.

## Datasets

Version 1 of the Zeitungs-LM was pretrained on the following publicly available datasets:

* [`biglam/europeana_newspapers`](https://huggingface.co/datasets/biglam/europeana_newspapers)
* [`storytracer/German-PD-Newspapers`](https://huggingface.co/datasets/storytracer/German-PD-Newspapers)

In total, the pretraining corpus has a size of 133GB.

## Benchmarks (Named Entity Recognition)

We compare our Zeitungs-LM directly to the Europeana BERT model (as Zeitungs-LM is supposed to be the successor of it) on various downstream tasks from the [hmBench](https://github.com/stefan-it/hmBench) repository, which is focussed on Named Entity Recognition.

Additionally, we use two additional datasets (ONB and LFT) from the ["A Named Entity Recognition Shootout for German"](https://aclanthology.org/P18-2020/) paper.

We report averaged micro F1-Score over 5 runs with different seeds and use the best hyper-parameter configuration on the development set of each dataset to report the final test score.

### Development Set

The results on the development set can be seen in the following table:

| Model \ Dataset     | [LFT][1]     | [ONB][2]     | [HisGermaNER][3] | [HIPE-2020][4] | [NewsEye][5] | [AjMC][6]    | [ZEFYS2025][7] | Avg.      |
|:--------------------|:-------------|:-------------|:-----------------|:---------------|:-------------|:-------------|:---------------|:----------|
| [Europeana BERT][8] | 79.66 ± 0.55 | 87.87 ± 0.62 | 81.60 ± 0.96     | 80.63 ± 0.42   | 41.21 ± 0.95 | 88.08 ± 0.81 | 86.52 ± 0.21   | 77.94     |
| [Zeitungs-LM v1][9] | 79.05 ± 0.32 | 88.45 ± 0.77 | 83.48 ± 1.15     | 81.32 ± 0.56   | 43.02 ± 1.12 | 89.50 ± 0.82 | 87.68 ± 0.41   | 78.93     |

Our Zeitungs-LM leads to a performance boost of 0.99% compared to the German Europeana BERT model.

### Test Set

The final results on the test set can be seen here:

| Model \ Dataset     | [LFT][1]     | [ONB][2]     | [HisGermaNER][3] | [HIPE-2020][4] | [NewsEye][5] | [AjMC][6]    | [ZEFYS2025][7] | Avg.      |
|:--------------------|:-------------|:-------------|:-----------------|:---------------|:-------------|:-------------|:---------------|:----------|
| [Europeana BERT][8] | 79.66 ± 0.55 | 84.80 ± 1.74 | 82.79 ± 1.01     | 77.62 ± 0.80   | 44.31 ± 3.04 | 90.59 ± 0.38 | 84.47 ± 0.49   | 77.75     |
| [Zeitungs-LM v1][9] | 79.49 ± 1.08 | 87.08 ± 0.30 | 84.51 ± 1.22     | 79.34 ± 0.96   | 45.97 ± 2.30 | 91.82 ± 1.72 | 87.17 ± 0.50   | 79.34     |

Our Zeitungs-LM beats the German Europeana BERT model by a large margin (1.59%).

[1]: https://aclanthology.org/P18-2020/
[2]: https://aclanthology.org/P18-2020/
[3]: https://huggingface.co/datasets/stefan-it/HisGermaNER
[4]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md
[5]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md
[6]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md
[7]: https://huggingface.co/datasets/SBB/ZEFYS2025
[8]: https://huggingface.co/dbmdz/bert-base-german-europeana-cased
[9]: https://huggingface.co/stefan-it/zeitungs-lm-v1

# Changelog

30.08.2026: Initial version of this repo with fresh evaluations on various Historical NER datasets for German.

# 🤖 AI disclosure

The repository introduces some "AI disclosure" rules:

* The README's are written 100% by a human. Aren't we all tired of Em-Dashes, and Oxford commata?
* The Python scripts always include an AI disclosure block, that is stating: a) which model was used, b) the AI-generated level and c) the level of human review.

Here's an example:

```python
"""<One-line description of what the file does.>

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   mostly        # fully | mostly | partially | none
    Human-Reviewed: partially     # fully | partially | minimally | none
"""
```

Currently, all Python scripts in this repository are written by a human (`AI-Generated: none`), but this will change in the future.

The rules are maintained as a standalone ruleset in my [AI Disclosure](https://github.com/stefan-it/ai-disclosure) repo.

# Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
