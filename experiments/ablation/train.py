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


def evaluate_model(eval_loader, model_env, mask=None):
    model = model_env.get_model()
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    metric = evaluate.load('accuracy')

    for eval_batch in eval_loader:
        eval_batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in eval_batch.items()}
        with torch.no_grad():
            outputs = model(**eval_batch, head_mask=mask.reshape(model_env.get_mask_shape())) if mask is not None else model(**eval_batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=eval_batch["labels"])

    return float(metric.compute()["accuracy"])


def test_shapley(checkpoint, dataset_name):
    print(f"=======CHECKPOINT: {checkpoint}==========")
    model_env = ModelEnvironment.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = StereotypeDataset.from_name(dataset_name, tokenizer)
    data_collator = DataCollatorWithPadding(model_env.get_tokenizer())
    eval_dataloader = DataLoader(dataset.get_eval_split(), shuffle=True, batch_size=512, collate_fn=data_collator)
    base_acc = evaluate_model(eval_dataloader, model_env)

    contribs = pull_contribs(checkpoint)

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


checkpoint = os.environ["CHECKPOINT"]
dataset_name = os.environ["DATASET"]


results = test_shapley(checkpoint, dataset_name)

print(results)

with open("results.json", "a") as file:
    file.write(json.dumps(results))

time = datetime.now()
api = HfApi()
api.upload_file(
    path_or_fileobj="results.json",
    path_in_repo=f"results.json",
    repo_id=checkpoint,
    repo_type="model",
)