"""Flair ColumnCorpus loader for the ZEFYS2025 dataset (SBB/ZEFYS2025 on the Hugging Face Hub).

AI Disclosure:
    Models:         Claude Fable 5
    AI-Generated:   mostly        # fully | mostly | partially | none
    Human-Reviewed: fully         # fully | partially | minimally | none
"""

import flair

from datasets import load_dataset

from flair.datasets.sequence_labeling import ColumnCorpus

from pathlib import Path
from typing import Optional, Union


class ZEFYS2025(ColumnCorpus):
    """ZEFYS2025: German NER dataset for historical newspapers (1837-1940), released by SBB.

    The dataset is hosted as parquet splits on the Hugging Face Hub. Each row holds a
    `tokens` list and a parallel `ner_tags` list (IOB2, classes PER/LOC/ORG). On first use
    every split is converted into a tab-separated CoNLL file (`train.txt`, `dev.txt`,
    `test.txt`) below the Flair dataset cache and loaded from there afterwards.
    """

    hf_repo_id = "SBB/ZEFYS2025"

    # Hugging Face split name -> Flair file name
    split_mapping = {"train": "train", "validation": "dev", "test": "test"}

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name

        column_format = {0: "text", 1: "ner"}

        for hf_split, flair_split in self.split_mapping.items():
            data_file = data_folder / f"{flair_split}.txt"

            if not data_file.is_file():
                self._write_conll_file(hf_split, data_file)

        super().__init__(
            data_folder,
            column_format,
            # A few tokens contain a space (e.g. "80 999"), so the file must be tab-separated.
            column_delimiter="\t",
            in_memory=in_memory,
            document_separator_token=None,
            **corpusargs,
        )

    def _write_conll_file(self, hf_split: str, data_file: Path) -> None:
        dataset = load_dataset(self.hf_repo_id, split=hf_split)

        data_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first so an interrupted download never leaves a
        # truncated split behind that would be picked up as complete on the next run.
        tmp_file = data_file.with_suffix(".tmp")

        with open(tmp_file, "w", encoding="utf-8") as f_out:
            for example in dataset:
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]

                if len(tokens) != len(ner_tags):
                    raise ValueError(
                        f"Token/label length mismatch in {self.hf_repo_id} ({hf_split}), id={example['id']}: "
                        f"{len(tokens)} tokens vs. {len(ner_tags)} tags"
                    )

                for token, ner_tag in zip(tokens, ner_tags):
                    f_out.write(f"{token}\t{ner_tag}\n")

                f_out.write("\n")

        tmp_file.rename(data_file)
