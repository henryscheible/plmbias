import json
import os

import evaluate
import numpy as np
import requests
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding
from datetime import datetime

from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment
import wandb

os.system(f"wandb login {os.environ.get('WANDB_TOKEN')}")

is_test = os.environ.get("IS_TEST") == "true"

if is_test:
    os.environ["CONTRIBS"] = "t5-small_stereoset_finetuned_contribs"
    os.environ["DATASET"] = "stereoset"

contribs_name = os.environ["CONTRIBS"]
dataset_name = os.environ["DATASET"]

run = wandb.init(project="plmbias", name=f"{contribs_name}_ablation")

def get_positive_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution > 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).to("cuda" if torch.cuda.is_available() else "cpu")


def get_negative_mask(contribs):
    ret = []
    for attribution in contribs:
        if attribution < 0:
            ret += [1]
        else:
            ret += [0]
    return torch.tensor(ret).to("cuda" if torch.cuda.is_available() else "cpu")


def get_bottom_up_masks(contribs):
    sorted_indices = np.argsort(contribs)
    masks = [np.zeros(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 1
        masks += [new_mask]
    return [torch.tensor(mask).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_top_down_masks(contribs):
    sorted_indices = np.argsort(contribs)
    masks = [np.ones(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 0
        masks += [new_mask]
    return [torch.tensor(mask).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]

def get_bottom_up_masks_rev(contribs):
    sorted_indices = np.argsort(-np.array(contribs))
    masks = [np.zeros(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 1
        masks += [new_mask]
    return [torch.tensor(mask).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def get_top_down_masks_rev(contribs):
    sorted_indices = np.argsort(-np.array(contribs))
    masks = [np.ones(len(contribs))]
    for i, index in enumerate(sorted_indices):
        new_mask = masks[i].copy()
        new_mask[index] = 0
        masks += [new_mask]
    return [torch.tensor(mask).to("cuda" if torch.cuda.is_available() else "cpu") for mask in masks]


def evaluate_model(eval_loader, model_env, mask=None):
    model = model_env.get_model()
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load('accuracy')

    for eval_batch in eval_loader:
        eval_batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in eval_batch.items()}
        model_env.evaluate_batch(eval_batch, mask, metric)

    return float(metric.compute()["accuracy"])


def test_shapley(contribs, model_env, dataset):
    print(f"=======CONTRIBS: {contribs_name}==========")

    
    data_collator = DataCollatorWithPadding(model_env.get_tokenizer())
    eval_dataloader = DataLoader(dataset.get_eval_split(), shuffle=True, batch_size=4096, collate_fn=data_collator)
    base_acc = evaluate_model(eval_dataloader, model_env)

    if not is_test:
        bottom_up_results = []
        for mask in tqdm(get_bottom_up_masks(contribs)):
            bottom_up_results += [evaluate_model(eval_dataloader, model_env, mask=mask)]

        top_down_results = []
        for mask in tqdm(get_top_down_masks(contribs)):
            top_down_results += [evaluate_model(eval_dataloader, model_env, mask=mask)]

        bottom_up_rev_results = []
        for mask in tqdm(get_bottom_up_masks_rev(contribs)):
            bottom_up_rev_results += [evaluate_model(eval_dataloader, model_env, mask=mask)]

        top_down_rev_results = []
        for mask in tqdm(get_top_down_masks_rev(contribs)):
            top_down_rev_results += [evaluate_model(eval_dataloader, model_env, mask=mask)]

        return {
            "base_acc": base_acc,
            "contribs": contribs,
            "bottom_up_results": list(bottom_up_results),
            "top_down_results": list(top_down_results),
            "bottom_up_rev_results": list(bottom_up_rev_results),
            "top_down_rev_results": list(top_down_rev_results)
        }
    else:
        return {
            "base_acc": base_acc,
            "contribs": contribs,
        }

contribs_artifact = run.use_artifact(contribs_name)
contribs_dir = contribs_artifact.download()
print(f"contribs_name: {contribs_name}, contribs_dir: {contribs_dir}")
with open(os.path.join(contribs_dir, "contribs.txt"), "r") as f:
    contribs = json.loads(f.read())
api = wandb.Api()
candidate_model_artifacts = filter(lambda x : x.type == "model", api.artifact(contribs_name).logged_by().used_artifacts())
model_artifact = list(candidate_model_artifacts)[0]
artifact_name = f"{model_artifact._project}/{model_artifact._artifact_name}"

artifact = run.use_artifact(artifact_name)
model_dir = artifact.download()

if "t5" in artifact_name:
    model_env = ModelEnvironment.from_pretrained_generative(model_dir)
    dataset = StereotypeDataset.from_name(dataset_name, model_env.get_tokenizer())
    model_env.setup_dataset(dataset)
else:
    model_env = ModelEnvironment.from_pretrained(model_dir)
    dataset = StereotypeDataset.from_name(dataset_name, model_env.get_tokenizer())

results = test_shapley(contribs, model_env, dataset)

print(results)

with open("results.json", "a") as file:
    file.write(json.dumps(results))

results_artifact = wandb.Artifact(name=f"{model_artifact._artifact_collection_name}_ablation", type="ablation_results")
results_artifact.add_file(local_path="results.json")
run.log_artifact(results_artifact)