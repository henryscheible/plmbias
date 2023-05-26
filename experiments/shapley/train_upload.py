import os

import evaluate
import numpy as np
import transformers
import torch
from captum.attr import ShapleyValueSampling
import json
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment
import wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

os.environ["WANDB_MODE"] = "online"

os.environ["CHECKPOINT"] = "t5-small_stereoset_finetuned"
os.environ["DATASET"] = "stereoset"


def compute_metrics_generative(logits, labels):
    true_label_id = model_env.get_tokenizer()("true").input_ids[0]
    false_label_id = model_env.get_tokenizer()("false").input_ids[0]
    true_logit = logits[:, 1, true_label_id]
    false_logit = logits[:, 1, false_label_id]
    binary_logits = np.stack([false_logit, true_logit], axis=-1)
    predictions = np.argmax(binary_logits, axis=-1)
    binary_labels = np.array(list(map(lambda label: 0 if label[0] == false_label_id else 1, labels)))
    confusion_matrix = np.zeros((2, 2))
    for label, pred in zip(binary_labels, predictions):
        confusion_matrix[label, pred] += 1
    return np.sum(predictions == binary_labels) / float(len(binary_labels))


def attribute_factory(model, eval_dataloader, shape):
    def attribute(mask):
        mask = mask.flatten()
        metric = evaluate.load("accuracy")
        model.eval()
        accuracies = []
        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            with torch.no_grad():
                if model_is_generative:
                    outputs = model(**eval_batch, head_mask=mask[:shape[0] * shape[1]].reshape(shape), decoder_head_mask=mask[shape[0] * shape[1]:].reshape(shape))
                else:
                    outputs = model(**eval_batch, head_mask=mask.reshape(shape))
            logits = outputs.logits
            labels = eval_batch["labels"]
            if model_is_generative:
                accuracies += [compute_metrics_generative(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())]
            else:
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=eval_batch["labels"])
        if model_is_generative:
            return np.mean(accuracies)
        else:
            return metric.compute()["accuracy"]

    return attribute

def get_shapley(eval_dataloader, model_env, num_samples=250, num_perturbations_per_eval=1):
    transformers.logging.set_verbosity_error()

    mask = torch.ones(model_env.get_mask_shape()).to(device).flatten().unsqueeze(0)
    if model_is_generative:
        mask = torch.ones(model_env.get_mask_shape()[0] * 2, model_env.get_mask_shape()[1]).to(device).flatten().unsqueeze(0)

    model = model_env.get_model().to(device)
    attribute = attribute_factory(model, eval_dataloader, model_env.get_mask_shape())

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            mask, n_samples=num_samples, show_progress=True,
            perturbations_per_eval=num_perturbations_per_eval
        )

    print(attribution)

    with open("contribs.txt", "w") as file:
        file.write(json.dumps(attribution.flatten().tolist()))

    return attribution


checkpoint = os.environ.get("CHECKPOINT")
dataset = os.environ.get("DATASET")

run = wandb.init(project="plmbias", name=f"{checkpoint}_contribs")


artifact_name = f"{checkpoint}:latest"
artifact = run.use_artifact(artifact_name)
model_dir = artifact.download()

# if "t5" in artifact_name:
#     model_is_generative = True
#     model_env = ModelEnvironment.from_pretrained_generative(model_dir)
#     dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
#     model_env.setup_dataset(dataset)
# else:
#     model_is_generative = False
#     model_env = ModelEnvironment.from_pretrained(model_dir)
#     dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
#
# data_collator = DataCollatorWithPadding(model_env.get_tokenizer())
# eval_dataloader = DataLoader(dataset.get_eval_split(), shuffle=True, batch_size=2048, collate_fn=data_collator)
# print("")
# get_shapley(eval_dataloader, model_env, num_samples=250)

contribs_artifact = wandb.Artifact(name=f"{checkpoint}_contribs", type="contribs")
contribs_artifact.add_file(local_path="contribs.txt")

run.log_artifact(contribs_artifact)