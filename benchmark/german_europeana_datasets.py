"""Flair ColumnCorpus loaders for the German Europeana NER datasets (LFT and ONB).

AI Disclosure:
    Models:         none
    AI-Generated:   none          # fully | mostly | partially | none
    Human-Reviewed: fully         # fully | partially | minimally | none
"""

import flair

from flair.datasets.sequence_labeling import ColumnCorpus

from huggingface_hub import hf_hub_download

from pathlib import Path
from typing import Optional, Union


class GERMAN_EUROPEANA_NER_LFT(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        column_format = {0: "text", 1: "ner"}

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        for dataset_file in ["enp_DE.lft.mr.tok.train.bio", "enp_DE.lft.mr.tok.dev.bio", "enp_DE.lft.mr.tok.test.bio"]:
            if not (data_folder / dataset_file).exists():
                # Download it from hub - ask @stefan-it for permission
                hf_path = hf_hub_download(repo_id="stefan-it/german-europeana-ner", repo_type="dataset",
                                          filename=dataset_file, token=True, local_dir=data_folder)

        super().__init__(
            data_folder,
            column_format=column_format,
            column_delimiter=" ",
            in_memory=in_memory,
            document_separator_token=None,
            **corpusargs,
        )


class GERMAN_EUROPEANA_NER_ONB(ColumnCorpus):
    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)

        column_format = {0: "text", 1: "ner"}

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        for dataset_file in ["enp_DE.onb.mr.tok.train.bio", "enp_DE.onb.mr.tok.dev.bio", "enp_DE.onb.mr.tok.test.bio"]:
            if not (data_folder / dataset_file).exists():
                # Download it from hub - ask @stefan-it for permission
                hf_path = hf_hub_download(repo_id="stefan-it/german-europeana-ner", repo_type="dataset",
                                          filename=dataset_file, token=True, local_dir=data_folder)

        super().__init__(
            data_folder,
            column_format=column_format,
            column_delimiter=" ",
            in_memory=in_memory,
            document_separator_token=None,
            **corpusargs,
        )
