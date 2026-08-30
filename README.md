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

For GPU experiments pick exactly one accelerator backend (they ship incompatible
PyTorch builds and are declared as mutually exclusive extras):

```bash
uv sync --extra cuda   # NVIDIA GPUs, stable wheels from the cu132 index
uv sync --extra xpu    # Intel GPUs, nightly wheels from the nightly/xpu index
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

| Model \ Dataset     | [LFT][1] | [ONB][2] | [HisGermaNER][3] | [HIPE-2020][4] | [NewsEye][5] | [AjMC][6] | Avg.      |
|:--------------------|:---------|:---------|:-----------------|:---------------|:-------------|:----------|:----------|
| [Europeana BERT][7] | 79.22    | 88.20    | 81.41            | 80.92          | 41.65        | 87.91     | 76.55     |
| Zeitungs-LM v1      | 79.39    | 88.53    | 83.10            | 81.55          | 44.53        | 89.71     | **77.80** |

Our Zeitungs-LM leads to a performance boost of 1.25% compared to the German Europeana BERT model.

### Test Set

The final results on the test set can be seen here:

| Model \ Dataset     | [LFT][1] | [ONB][2] | [HisGermaNER][3] | [HIPE-2020][4] | [NewsEye][5] | [AjMC][6] | Avg.
|:--------------------|:---------|:---------|:-----------------|:---------------|:-------------|:----------|:---------|
| [Europeana BERT][7] | 80.43    | 84.39    | 83.21            | 77.49          | 42.96        | 90.52     | 76.50    |
| Zeitungs-LM v1      | 80.35    | 87.28    | 84.92            | 79.91          | 47.16        | 92.76     | **78.73**|

Our Zeitungs-LM beats the German Europeana BERT model by a large margin (2.23%).

[1]: https://aclanthology.org/P18-2020/
[2]: https://aclanthology.org/P18-2020/
[3]: https://huggingface.co/datasets/stefan-it/HisGermaNER
[4]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md
[5]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md
[6]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md
[7]: https://huggingface.co/dbmdz/bert-base-german-europeana-cased

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
