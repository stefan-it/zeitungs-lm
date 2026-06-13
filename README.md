# Zeitungs-LM

This repository hosts all necessary information for benchmarking the [Zeitungs-LM](https://huggingface.co/stefan-it/zeitungs-lm-v1), a language model trained on Historical German Newspapers.

Technically the model itself is an ELECTRA model, which was pretrained with the [TEAMS](https://aclanthology.org/2021.findings-acl.219/) approach.

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

13.06.2026: Initial version of this repo with fresh evaluations on various Historical NER datasets for German.

# Acknowledgements

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
