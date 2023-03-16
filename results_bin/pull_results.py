from itertools import product

import pandas as pd
import requests
import urllib.request



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



def pull_results():
    names = [f"{model}_{dataset}_{train_type}" for model, dataset, train_type in product(models, datasets, training_types)]
    for name in names:
        urllib.request.urlretrieve(f"https://huggingface.co/henryscheible/{name}/resolve/main/heatmap.pdf", f"{name}_heatmap.pdf")
        urllib.request.urlretrieve(f"https://huggingface.co/henryscheible/{name}/resolve/main/add_ablation.pdf", f"{name}_add_ablation.pdf")
        urllib.request.urlretrieve(f"https://huggingface.co/henryscheible/{name}/resolve/main/remove_ablation.pdf", f"{name}_remove_ablation.pdf")
        urllib.request.urlretrieve(f"https://huggingface.co/henryscheible/{name}/resolve/main/ss_ablation.pdf", f"{name}_ss_ablation.pdf")
        urllib.request.urlretrieve(f"https://huggingface.co/henryscheible/{name}/resolve/main/ss_ablation_stats.json", f"{name}_ss_ablation_stats.json")



if __name__ == "__main__":
    pull_results()