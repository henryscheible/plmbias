from itertools import cycle, product
import json

import requests

contexts = {
    "dsail2": "ssh://henry@dsail2.cs.dartmouth.edu",
    "mms-large-1": "ssh://henry@mms-large-1.cs.dartmouth.edu",
    "mms-large-2": "ssh://henry@mms-large-2.cs.dartmouth.edu",
}

models = [
    "bert-large-uncased",
    "gpt2",
    "xlnet-base-cased",
]

datasets = [
    "stereoset",
    "winobias",
    "crows_pairs"
]

training_types = [
    # "classifieronly",
    "finetuned"
]

gpu_cards = [
    ("mms-large-1", 0),
    # ("mms-large-2", 0),
    ("dsail2", 0),
    ("mms-large-1", 1),
    # ("mms-large-2", 1),
    ("dsail2", 1),
    ("mms-large-1", 2),
    # ("mms-large-2", 2),
    ("dsail2", 2),
    # ("mms-large-2", 3),
    ("dsail2", 3),
    ("mms-large-1", 4),
    # ("mms-large-2", 4),
    ("mms-large-1", 5),
    # ("mms-large-2", 5),
    ("mms-large-1", 6),
    # ("mms-large-2", 6),
    ("mms-large-1", 7),
    # ("mms-large-2", 7),
]

learning_rates = [
    5e-5, 1e-4, 5e-4, 1e-3, 5e-3
]

config = dict()
config["contexts"] = contexts
config["experiments"] = []


def not_already_trained(checkpoint):
    model, dataset, training_type, _ = checkpoint
    try:
        validation = requests.get(f"https://huggingface.co/henryscheible/{model}_{dataset}_{training_type}/raw/main/README.md").text
        val_lines = validation.split("\n")
        acc_line = list(filter(lambda l: "Accuracy:" in l, val_lines))[0]
        acc = acc_line[12:]
        return float(acc) < 0.7
    except:
        return True


configs = product(models, datasets, training_types, learning_rates)
required_configs = filter(not_already_trained, configs)

for (model, dataset, training_type, lr), (context, card) in zip(required_configs, cycle(gpu_cards)):
    config["experiments"].append({
      "name": f"{model}_{dataset}_{training_type}_{lr}",
      "image": "train",
      "context": context,
      "card": card,
      "buildargs": {
        "MODEL": model,
        "DATASET": dataset,
        "TRAIN_TYPE": training_type,
        "LR": 5e-5
      }
    })

with open("training_lr_sweep.json", "w") as f:
    f.write(json.dumps(config))
