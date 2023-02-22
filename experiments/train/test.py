import os

os.environ["HF_TOKEN"] = "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"
os.environ["MODEL"] = "bert-base-uncased"
os.environ["TRAIN_TYPE"] = "classifieronly"
os.environ["DATASET"] = "stereoset"

from huggingface_hub import HfFolder
HfFolder.save_token(os.environ.get("HF_TOKEN"))

from experiments.train import train

