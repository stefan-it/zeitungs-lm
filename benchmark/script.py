"""Benchmark runner: reads a JSON config and runs the seed/hyper-parameter grid via experiment.py.

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   partially     # fully | mostly | partially | none
    Human-Reviewed: fully         # fully | partially | minimally | none
"""

import flair
import json
import os
import torch

from experiment import ExperimentConfiguration, get_output_path, run_experiment

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

# Optional device override (e.g. "cuda:1" or "cpu"). If absent, Flair picks
# CUDA when available and falls back to CPU otherwise.
device = json_config.get("device")
if device is not None:
    flair.device = torch.device(device)

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

                        # Skip hyper-parameter configs that already have an output folder
                        output_path = get_output_path(experiment_configuration)
                        if Path(output_path).exists():
                            print(f"Skipping experiment, output folder already exists: {output_path}")
                            continue

                        output_path = run_experiment(experiment_configuration=experiment_configuration)
