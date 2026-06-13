import flair
import json
import os

from experiment import ExperimentConfiguration, run_experiment

from huggingface_hub import login, HfApi

from pathlib import Path

# Hugging Face Model Hub configuration
config_file     = os.environ.get("CONFIG")
hf_token        = os.environ.get("HF_TOKEN")

login(token=hf_token, add_to_git_credential=True)
api = HfApi()

with open(config_file, "rt") as f_p:
    json_config = json.load(f_p)

seeds            = json_config["seeds"]
batch_sizes      = json_config["batch_sizes"]
epochs           = json_config["epochs"]
learning_rates   = json_config["learning_rates"]
subword_poolings = json_config["subword_poolings"]
context_sizes    = json_config["context_sizes"]
hf_model         = json_config["hf_model"]
model_short_name = json_config["model_short_name"]
task             = json_config["task"]

device = json_config["device"]
flair.device = device

for seed in seeds:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for learning_rate in learning_rates:
                for subword_pooling in subword_poolings:
                    for context_size in context_sizes:
                        experiment_configuration = ExperimentConfiguration(
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            epoch=epoch,
                            context_size=context_size,
                            seed=seed,
                            subtoken_pooling=subword_pooling,
                            base_model=hf_model,
                            base_model_short=model_short_name,
                            task=task,
                        )
                        output_path = run_experiment(experiment_configuration=experiment_configuration)
