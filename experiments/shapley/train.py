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

is_test = os.environ.get("IS_TEST") == "true"

if is_test:
    os.environ["CHECKPOINT"] = "t5-small_stereoset_finetuned"
    os.environ["DATASET"] = "stereoset"
    os.environ["SOURCE"] = "wandb"
    os.environ["SAMPLES"] = "1"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def attribute_factory(model, eval_dataloader, portion = None):
    def attribute(mask):
        if is_test:
            if model_env.has_evaled:
                return 0
        mask = mask.flatten()
        metric = evaluate.load("accuracy")
        model.eval()
        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            if portion == "encoder":
                model_env.evaluate_batch(eval_batch, metric, encoder_mask = mask, decoder_mask = torch.ones(model_env.get_mask_shape_decoder()).to(device))
            if portion == "decoder":
                model_env.evaluate_batch(eval_batch, metric, decoder_mask = mask, encoder_mask = torch.ones(model_env.get_mask_shape()).to(device))
            else:
                model_env.evaluate_batch(eval_batch, metric, mask)
        if is_test:
            model_env.has_evaled = True
        return metric.compute()["accuracy"]

    return attribute

def get_shapley(eval_dataloader, model_env, num_samples=250, num_perturbations_per_eval=1):
    transformers.logging.set_verbosity_error()
    model = model_env.get_model().to(device)


    mask = torch.ones(model_env.get_mask_shape()).to(device).flatten().unsqueeze(0)
    attribute = attribute_factory(model, eval_dataloader, model_env.get_mask_shape())
    mask = torch.ones(model_env.get_mask_shape()).to(device).flatten().unsqueeze(0)


    if model_is_generative:
        dec_mask = torch.ones(model_env.get_mask_shape_decoder()).to(device).flatten().unsqueeze(0)
        enc_attribute = attribute_factory(model, eval_dataloader, portion="encoder")
        dec_attribute = attribute_factory(model, eval_dataloader, portion="decoder")
        with torch.no_grad():
            model.eval()
            sv = ShapleyValueSampling(enc_attribute)
            enc_attribution = sv.attribute(
                mask, n_samples=num_samples, show_progress=True,
                perturbations_per_eval=num_perturbations_per_eval
            )
            model_env.has_evaled = False
            sv = ShapleyValueSampling(dec_attribute)
            dec_attribution = sv.attribute(
                dec_mask, n_samples=num_samples, show_progress=True,
                perturbations_per_eval=num_perturbations_per_eval
            )
        print("ENC CONTRIBS")
        print(enc_attribution)
        print("DEC CONTRIBS")
        print(dec_attribution)

        with open("enc_contribs.txt", "w") as file:
            file.write(json.dumps(enc_attribution.flatten().tolist()))
        with open("dec_contribs.txt", "w") as file:
            file.write(json.dumps(dec_attribution.flatten().tolist()))
    else:
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



checkpoint = os.environ.get("CHECKPOINT")
dataset = os.environ.get("DATASET")
source = os.environ.get("SOURCE")

run = wandb.init(project="plmbias", name=f"{checkpoint}_contribs")

if source == "wandb":
    artifact_name = f"model-{checkpoint}:latest"
    artifact = run.use_artifact(artifact_name)
    model_dir = artifact.download()
else:
    model_dir = checkpoint

if "t5" in artifact_name:
    model_is_generative = True
    model_env = ModelEnvironment.from_pretrained_generative(model_dir)
    dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
    model_env.setup_dataset(dataset)
else:
    model_is_generative = False
    model_env = ModelEnvironment.from_pretrained(model_dir)
    dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())

model_env.has_evaled = False


data_collator = DataCollatorWithPadding(model_env.get_tokenizer())
eval_dataloader = DataLoader(dataset.get_eval_split(), shuffle=True, batch_size=2048, collate_fn=data_collator)
print("")
num_samples = os.environ.get("SAMPLES")
num_samples = int(num_samples) if num_samples is not None else 1
get_shapley(eval_dataloader, model_env, num_samples=num_samples)

if source == "wandb":
    contribs_artifact = wandb.Artifact(name=f"{checkpoint}_contribs", type="contribs")
    if model_is_generative:
        contribs_artifact.add_file(local_path="enc_contribs.txt")
        contribs_artifact.add_file(local_path="dec_contribs.txt")
    else:
        contribs_artifact.add_file(local_path="contribs.txt")
    run.log_artifact(contribs_artifact)