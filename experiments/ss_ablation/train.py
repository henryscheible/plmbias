import json

import os
import numpy as np
import requests
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datetime import datetime

import wandb

from plmbias.models import ModelEnvironment
from plmbias.stereotypescore import StereotypeScoreCalculator

is_test = os.environ.get("IS_TEST") == "true"

if is_test:
    os.environ["CONTRIBS"] = "t5-small_stereoset_finetuned_contribs:latest"
    os.environ["MODEL"] = "t5-small"

contribs_name = os.environ["CONTRIBS"]
model_name = os.environ["MODEL"]
spec = "_".join(contribs_name.split("_")[:-1])

run = wandb.init(project="plmbias" if not is_test else "plmbias-test", name=f"{spec}_ss-ablation")


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


def get_ss(calc, model_env, mask=None):
    model = model_env.get_model()
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if mask is not None:
        calc.set_intersentence_mask(mask)
    calc.set_intersentence_model(model)
    return calc()["intersentence"]


def test_shapley(contribs, model_env, is_generative=False):
    tokenizer = model_env.get_tokenizer()
    calc = StereotypeScoreCalculator(model_env, tokenizer, model_env, tokenizer, is_generative=is_generative)
    base_lm, base_ss = get_ss(calc, model_env)

    print("CALCULATING BOTTOM UP")


    bottom_up_ss = []
    bottom_up_lm = []
    for mask in tqdm(get_bottom_up_masks(contribs)):
        ss, lm = get_ss(calc, model_env, mask=mask)
        bottom_up_ss += [ss]
        bottom_up_lm += [lm]
        if is_test:
            break

    print("CALCULATING TOP DOWN")

    top_down_ss = []
    top_down_lm = []
    if not is_test:
        for mask in tqdm(get_top_down_masks(contribs)):
            ss, lm = get_ss(calc, model_env, mask=mask)
            top_down_ss += [ss]
            top_down_lm += [ss]

    print("CALCULATING BOTTOM_UP_REV")

    bottom_up_rev_ss = []
    bottom_up_rev_lm = []
    if not is_test:
        for mask in tqdm(get_bottom_up_masks_rev(contribs)):
            ss, lm = get_ss(calc, model_env, mask=mask)
            bottom_up_rev_ss += [ss]
            bottom_up_rev_lm += [ss]

    print("CALCULATING TOP_DOWN_REV")

    top_down_rev_ss = []
    top_down_rev_lm = []
    if not is_test:
        for mask in tqdm(get_top_down_masks_rev(contribs)):
            ss, lm = get_ss(calc, model_env, mask=mask)
            top_down_rev_ss += [ss]
            top_down_rev_lm += [ss]

    return {
        "base_ss": base_ss,
        "base_lm": base_lm,
        "contribs": contribs,
        "bottom_up_ss": list(bottom_up_ss),
        "bottom_up_lm": list(bottom_up_lm),
        "top_down_ss": list(top_down_ss),
        "top_down_lm": list(top_down_lm),
        "bottom_up_rev_ss": list(bottom_up_rev_ss),
        "bottom_up_rev_lm": list(bottom_up_rev_lm),
        "top_down_rev_ss": list(top_down_rev_ss),
        "top_down_rev_lm": list(top_down_rev_lm),
    }

project = "plmbias" if not is_test else "plmbias-test"
contribs_artifact = run.use_artifact(f"{contribs_name}")
contribs_dir = contribs_artifact.download()
print(f"contribs_name: {contribs_name}, contribs_dir: {contribs_dir}")
with open(os.path.join(contribs_dir, f"enc_contribs.txt"), "r") as f:
    contribs = json.loads(f.read())
api = wandb.Api()
candidate_model_artifacts = filter(lambda x : x.type == "model", api.artifact(f"{project}/{contribs_name}").logged_by().used_artifacts())
model_artifact = list(candidate_model_artifacts)[0]
artifact_name = f"{model_artifact._project}/{model_artifact._artifact_name}"

if "t5" in artifact_name:
    model_env = ModelEnvironment.from_pretrained_generative(model_name)
else:
    model_env = ModelEnvironment.from_pretrained(model_name)


results = test_shapley(contribs, model_env, is_generative="t5" in artifact_name)

print(results)

with open("results.json", "a") as file:
    file.write(json.dumps(results))

results_artifact = wandb.Artifact(name=f"{model_artifact._artifact_collection_name}_{portion}_ss_ablation", type="ss_ablation_results")
results_artifact.add_file(local_path="results.json")
run.log_artifact(results_artifact)