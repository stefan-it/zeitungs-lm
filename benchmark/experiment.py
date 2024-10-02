import logging

from dataclasses import dataclass

from flair import set_seed

from flair.datasets import  NER_HIPE_2022
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

from german_europeana_datasets import GERMAN_EUROPEANA_NER_LFT, GERMAN_EUROPEANA_NER_ONB
from hisgermaner_dataset import HisGermaNER
from utils import prepare_ajmc_corpus, prepare_clef_2020_corpus, prepare_newseye_de_fr_corpus

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ExperimentConfiguration:
    batch_size: int
    learning_rate: float
    epoch: int
    context_size: int
    seed: int
    base_model: str
    base_model_short: str
    task: str
    layers: str = "-1"
    subtoken_pooling: str = "first"
    use_crf: bool = False
    use_tensorboard: bool = True


def run_experiment(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    # Possible task names:
    # - ner/german_europeana_onb
    # - ner/german_europeana_lft
    # - ner/hipe2022_ajmc_de
    # - ner/hipe2020
    # - ner/hisgermaner
    # - ner/newseye

    label_type = None

    if experiment_configuration.task.startswith("ner"):
        label_type = "ner"

        corpus = None

        if experiment_configuration.task == "ner/german_europeana_onb":
            corpus = GERMAN_EUROPEANA_NER_ONB()
        elif experiment_configuration.task == "ner/german_europeana_lft":
            corpus = GERMAN_EUROPEANA_NER_LFT()
        elif experiment_configuration.task == "ner/hipe2022_ajmc_de":
            preproc_fn = prepare_ajmc_corpus
            corpus = NER_HIPE_2022(dataset_name="ajmc", language="de", preproc_fn=preproc_fn,
                                   add_document_separator=True)
        elif experiment_configuration.task == "ner/hipe2020":
            preproc_fn = prepare_clef_2020_corpus
            corpus = NER_HIPE_2022(dataset_name="hipe2020", language="de", preproc_fn=preproc_fn,
                                   add_document_separator=True)
        elif experiment_configuration.task == "ner/newseye":
            preproc_fn = prepare_newseye_de_fr_corpus
            corpus = NER_HIPE_2022(dataset_name="newseye", language="de", preproc_fn=preproc_fn,
                                   add_document_separator=True)
        elif experiment_configuration.task == "ner/hisgermaner":
            corpus = HisGermaNER()

        label_dictionary = corpus.make_label_dictionary(label_type=label_type)
        logger.info("Label Dictionary: {}".format(label_dictionary.get_items()))

        embeddings = TransformerWordEmbeddings(
            model=experiment_configuration.base_model,
            layers=experiment_configuration.layers,
            subtoken_pooling=experiment_configuration.subtoken_pooling,
            fine_tune=True,
            use_context=experiment_configuration.context_size,
        )

        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=label_dictionary,
            tag_type=label_type,
            use_crf=experiment_configuration.use_crf,
            use_rnn=False,
            reproject_embeddings=False,
        )

        trainer = ModelTrainer(tagger, corpus)

        output_path_parts = [
            "flair",
            experiment_configuration.task.split("/")[0],
            experiment_configuration.task.split("/")[1].replace("_", "-"),
            experiment_configuration.base_model_short,
            f"bs{experiment_configuration.batch_size}",
            f"e{experiment_configuration.epoch}",
            f"lr{experiment_configuration.learning_rate}",
            str(experiment_configuration.seed)
        ]

        output_path = "-".join(output_path_parts)

        plugins = []

        if experiment_configuration.use_tensorboard:
            logger.info("TensorBoard logging is enabled")

            tb_path = Path(f"{output_path}/runs")
            tb_path.mkdir(parents=True, exist_ok=True)

            plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

        trainer.fine_tune(
            output_path,
            learning_rate=experiment_configuration.learning_rate,
            mini_batch_size=experiment_configuration.batch_size,
            max_epochs=experiment_configuration.epoch,
            shuffle=True,
            embeddings_storage_mode='none',
            weight_decay=0.,
            use_final_model_for_eval=False,
            plugins=plugins,
        )

        # Finally, print model card for information
        tagger.print_model_card()

        return output_path
