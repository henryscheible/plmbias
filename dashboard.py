from itertools import product

import pandas as pd
import requests
from flask import Flask
from pretty_html_table import build_table


app = Flask(__name__)


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



@app.route('/')
def hello():
    results = pd.DataFrame(columns=["name", "accuracy", "contribs", "ablation", "ss_ablation", "heatmap", "add_ablation_figure", "remove_ablation_figure", "ss_ablation_stats", "ss_ablation_figure"])
    names = [f"{model}_{dataset}_{train_type}" for model, dataset, train_type in product(models, datasets, training_types)]
    results["name"] = names
    results = results.set_index("name")
    for name in names:
        try:
            validation = requests.get(f"https://huggingface.co/henryscheible/{name}/raw/main/README.md").text
            val_lines = validation.split("\n")
            acc_line = list(filter(lambda l: "Accuracy:" in l, val_lines))[0]
            acc = acc_line[12:]
            results.loc[name]["accuracy"] = acc
        except:
            pass
        try:
            contribs = requests.get(
                f"https://huggingface.co/henryscheible/{name}/raw/main/contribs.txt").text
            if contribs[0] == "[":
                results.loc[name]["contribs"] = True
        except:
            pass
        try:
            ablation = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/results.json").text
            if "Entry not found" not in ablation:
                results.loc[name]["ablation"] = True
        except:
            pass
        try:
            ss_ablation = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/ss_ablation_results.json").text
            if "Entry not found" not in ss_ablation:
                results.loc[name]["ss_ablation"] = True
        except:
            pass
        try:
            heatmap = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/heatmap.pdf").text
            if "Entry not found" not in heatmap:
                results.loc[name]["heatmap"] = True
        except:
            pass
        try:
            add_ablation_figure = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/add_ablation.pdf").text
            if "Entry not found" not in add_ablation_figure:
                results.loc[name]["add_ablation_figure"] = True
        except:
            pass
        try:
            ss_ablation_figure = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/ss_ablation.pdf").text
            if "Entry not found" not in ss_ablation_figure:
                results.loc[name]["ss_ablation_figure"] = True
        except:
            pass
        try:
            ss_ablation_stats = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/ss_ablation_stats.json").text
            if "Entry not found" not in ss_ablation_stats:
                results.loc[name]["ss_ablation_stats"] = True
        except:
            pass
        try:
            remove_ablation_figure = requests.get(f"https://huggingface.co/henryscheible/{name}/blob/main/remove_ablation.pdf").text
            if "Entry not found" not in remove_ablation_figure:
                results.loc[name]["remove_ablation_figure"] = True
        except:
            pass
    results.reset_index(inplace=True)

    return build_table(results, 'blue_light')