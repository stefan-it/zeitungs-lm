import json

from pathlib import Path

checkpoints = [model_name for model_name in Path("./").iterdir() if model_name.name.startswith("ckpt-")]

german_europeana_lft_config_template = \
    """{
    "batch_sizes": [
        16,
        8
    ],
    "learning_rates": [
        3e-5,
        5e-5
    ],
    "epochs": [
        10
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "HF_MODEL",
    "model_short_name": "MODEL_SHORT_NAME",
    "task": "ner/german_europeana_lft",
    "cuda": "0"
}"""

german_europeana_onb_config_template = \
    """{
    "batch_sizes": [
        16,
        8
    ],
    "learning_rates": [
        3e-5,
        5e-5
    ],
    "epochs": [
        10
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "HF_MODEL",
    "model_short_name": "MODEL_SHORT_NAME",
    "task": "ner/german_europeana_onb",
    "cuda": "0"
}
"""

hisgermaner_config_template = \
    """{
    "batch_sizes": [
        4,
        8
    ],
    "learning_rates": [
        3e-5,
        5e-5
    ],
    "epochs": [
        10
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "HF_MODEL",
    "model_short_name": "MODEL_SHORT_NAME",
    "task": "ner/hisgermaner",
    "cuda": "0"
}
"""

hipe2020_config_template = \
    """{
    "batch_sizes": [
        8,
        4
    ],
    "learning_rates": [
        3e-5,
        5e-5
    ],
    "epochs": [
        10
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "HF_MODEL",
    "model_short_name": "MODEL_SHORT_NAME",
    "task": "ner/hipe2020",
    "cuda": "0"
}
"""

newseye_config_template = \
    """{
    "batch_sizes": [
        8,
        4
    ],
    "learning_rates": [
        3e-5,
        5e-5
    ],
    "epochs": [
        10
    ],
    "context_sizes": [
        0
    ],
    "seeds": [
        1,
        2,
        3,
        4,
        5
    ],
    "layers": "-1",
    "subword_poolings": [
        "first"
    ],
    "use_crf": false,
    "use_tensorboard": true,
    "hf_model": "HF_MODEL",
    "model_short_name": "MODEL_SHORT_NAME",
    "task": "ner/newseye",
    "cuda": "0"
}
"""

config_templates = {
    "german_europeana_lft": german_europeana_lft_config_template,
    "german_europeana_onb": german_europeana_onb_config_template,
    "hisgermaner": hisgermaner_config_template,
    "hipe2020": hipe2020_config_template,
    "newseye": newseye_config_template,
}

models = [
    {
        "hf_model": "stefan-it/zeitungs-lm-v1",
        "model_short_name": "zeitungs_lm_v1",
    },
    {
        "hf_model": "dbmdz/bert-base-german-europeana-cased",
        "model_short_name": "europeana_bert",
    },
]

for model in models:
    hf_model = model["hf_model"]
    model_short_name = model["model_short_name"]
    print("Create config for model:", f"'{hf_model}'", "known as", f"'{model_short_name}'")

    # First, create folder
    p = Path(model_short_name)
    p.mkdir(parents=True, exist_ok=True)

    for config_name, config_template in config_templates.items():
        config_template = config_template.replace("HF_MODEL", hf_model)
        config_template = config_template.replace("MODEL_SHORT_NAME", model_short_name)

        with open(p / f"{config_name}.json", "wt") as f_out:
            f_out.write(config_template + "\n")
