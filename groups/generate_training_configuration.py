from itertools import cycle, product
import json

contexts = {
    "dsail2": "ssh://henry@dsail2.cs.dartmouth.edu",
    "mms-large-1": "ssh://henry@mms-large-1.cs.dartmouth.edu",
    "mms-large-2": "ssh://henry@mms-large-2.cs.dartmouth.edu",
}

models = [
    "bert-base-large",
    "gpt2",
    "xlnet"
]

datasets = [
    "stereoset",
    "winobias",
    "crows_pairs"
]

training_types = [
    "classifieronly",
    "finetuned"
]

gpu_cards = [
    ("mms-large-1", 0),
    ("mms-large-1", 1),
    ("mms-large-1", 2),
    ("mms-large-1", 3),
    ("mms-large-1", 4),
    ("mms-large-1", 5),
    ("mms-large-1", 6),
    ("mms-large-1", 7),
    ("dsail2", 0),
    ("dsail2", 1),
    ("dsail2", 2),
    ("dsail2", 3),
    ("mms-large-2", 0),
    ("mms-large-2", 1),
    ("mms-large-2", 2),
    ("mms-large-2", 3),
    ("mms-large-2", 4),
    ("mms-large-2", 5),
    ("mms-large-2", 6),
    ("mms-large-2", 7),
]

config = dict()
config["contexts"] = contexts
config["experiments"] = []
for (model, dataset, training_type), (context, card) in zip(product(models, datasets, training_types), cycle(gpu_cards)):
    config["experiments"].append({
      "name": f"{model}_{dataset}_{training_type}",
      "image": "train",
      "context": context,
      "card": card,
      "buildargs": {
        "MODEL": model,
        "DATASET": dataset,
        "TRAIN_TYPE": training_type
      }
    })

with open("training.json", "w") as f:
    f.write(json.dumps(config))
