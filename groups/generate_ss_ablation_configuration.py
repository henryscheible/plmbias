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
    # ("mms-large-1", 6),
    # ("mms-large-2", 6),
    # ("mms-large-1", 7),
    # ("mms-large-2", 7),
]
config = dict()
config["contexts"] = contexts
config["experiments"] = []


def has_already_trained(checkpoint):
    model, dataset, training_type = checkpoint
    try:
        validation = requests.get(f"https://huggingface.co/henryscheible/{model}_{dataset}_{training_type}/raw/main/README.md").text
        val_lines = validation.split("\n")
        acc_line = list(filter(lambda l: "Accuracy:" in l, val_lines))[0]
        acc = acc_line[12:]
        return float(acc) > 0.7
    except:
        return False


def has_already_probed(checkpoint):
    model, dataset, training_type = checkpoint
    try:
        probing = requests.get(f"https://huggingface.co/henryscheible/{model}_{dataset}_{training_type}/blob/main/contribs.txt").text
        if "Entry not found" not in probing:
            return True
    except:
        return False


def has_already_ablated(checkpoint):
    model, dataset, training_type = checkpoint
    try:
        ablation = requests.get(f"https://huggingface.co/henryscheible/{model}_{dataset}_{training_type}/blob/main/ss_ablation_results.json").text
        if "Entry not found" not in ablation:
            return True
    except:
        return False


def needs_ablating(checkpoint):
    return has_already_trained(checkpoint) and has_already_probed(checkpoint) #and not has_already_ablated(checkpoint)


configs = product(models, datasets, training_types)
required_configs = filter(needs_ablating, configs)

for (model, dataset, training_type), (context, card) in zip(required_configs, cycle(gpu_cards)):
    config["experiments"].append({
      "name": f"{model}_{dataset}_{training_type}_ss_ablation_v2",
      "image": "ss_ablation",
      "context": context,
      "card": card,
      "buildargs": {
        "CHECKPOINT": f"henryscheible/{model}_{dataset}_{training_type}",
        "DATASET": dataset
      }
    })

with open("ss_ablation3.json", "w") as f:
    f.write(json.dumps(config))
