import json

import os
import numpy as np
import requests
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from datetime import datetime

from plmbias.models import ModelEnvironment
from plmbias.stereotypescore import StereotypeScoreCalculator


def pull_contribs(checkpoint):
    res = requests.get(f"https://huggingface.co/{checkpoint}/raw/main/contribs.txt")
    print(f"https://huggingface.co/{checkpoint}/raw/main/contribs.txt")
    print(res.text)
    return json.loads(res.text)


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
        calc.set_intrasentence_mask(mask)
    calc.set_intrasentence_model(model)
    return calc()["intrasentence"]


def test_shapley(checkpoint):
    print(f"=======CHECKPOINT: {checkpoint}==========")
    model_env = ModelEnvironment.from_pretrained_lm(checkpoint)
    tokenizer = model_env.get_tokenizer()
    tokenizer.mask_token = "[MASK]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.cls_token = "[CLS]"
    calc = StereotypeScoreCalculator(model_env, tokenizer, model_env, tokenizer)
    base_lm, base_ss = get_ss(calc, model_env)

    contribs = pull_contribs(checkpoint)

    print("CALCULATING BOTTOM UP")

    bottom_up_ss = []
    bottom_up_lm = []
    for mask in tqdm(get_bottom_up_masks(contribs)):
        ss, lm = get_ss(calc, model_env, mask=mask)
        bottom_up_ss += [ss]
        bottom_up_lm += [lm]

    print("CALCULATING TOP DOWN")

    top_down_ss = []
    top_down_lm = []
    for mask in tqdm(get_top_down_masks(contribs)):
        ss, lm = get_ss(calc, model_env, mask=mask)
        top_down_ss += [ss]
        top_down_lm += [ss]

    print("CALCULATING BOTTOM_UP_REV")

    bottom_up_rev_ss = []
    bottom_up_rev_lm = []
    for mask in tqdm(get_bottom_up_masks_rev(contribs)):
        ss, lm = get_ss(calc, model_env, mask=mask)
        bottom_up_rev_ss += [ss]
        bottom_up_rev_lm += [ss]

    print("CALCULATING TOP_DOWN_REV")

    top_down_rev_ss = []
    top_down_rev_lm = []
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


checkpoint = os.environ["CHECKPOINT"]

results = test_shapley(checkpoint)

print(results)

with open("results.json", "a") as file:
    file.write(json.dumps(results))

time = datetime.now()
api = HfApi()
api.upload_file(
    path_or_fileobj="results.json",
    path_in_repo=f"ss_ablation_results.json",
    repo_id=checkpoint,
    repo_type="model",
)