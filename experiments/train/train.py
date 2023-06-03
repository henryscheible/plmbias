import os
import datetime
import string
from typing import Dict, Union, Any, Optional, List, Tuple

import evaluate
import numpy as np
import torch
import wandb
import random
import transformers
from torch import nn
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments

from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),


os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_WATCH"] = "all"

is_test = os.environ.get("IS_TEST") == "true"

if is_test:
    os.environ["MODEL"] = "t5-small"
    os.environ["DATASET"] = "stereoset"
    os.environ["TRAIN_TYPE"] = "finetuned"
    os.environ["LEARNING_RATE"] = "5e-4"
    os.environ["MODEL_TYPE"] = "generative"
    os.environ["WANDB_MODE"] = "offline"

config = dict()


rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


transformers.set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
transformers.enable_full_determinism(42)

config = wandb.config
hf_model_id = os.environ.get("MODEL")
learning_rate = float(os.environ.get("LEARNING_RATE")) if os.environ.get("LEARNING_RATE") is not None else (5e-4 if "t5" in hf_model_id else 5e-5)
train_type = os.environ.get("TRAIN_TYPE")
dataset = os.environ.get("DATASET")
model_type = os.environ.get("MODEL_TYPE")
group = f"{hf_model_id.replace('/', '-')}_{dataset}_{train_type}"
name = f"{group}_{rand_id}"
run = wandb.init(name=group, project="plmbias", group="train")
wandb.define_metric("eval/accuracy", summary="max")

if model_type == "generative":
    model_env = ModelEnvironment.from_pretrained_generative(hf_model_id)
    dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer(), is_generative=True)
    model_env.setup_dataset(dataset)
elif model_type == "causal":
    model_env = ModelEnvironment.from_pretrained_causal(hf_model_id)
    dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
else:
    model_env = ModelEnvironment.from_pretrained(hf_model_id)
    dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())

data_collator = DataCollatorWithPadding(model_env.get_tokenizer())

model_env.get_model().to(device)

os.environ["WANDB_PROJECT"] = name

compute_metrics = model_env.get_compute_metrics_fn()

for param in model_env.get_model().parameters():
    param.requires_grad = False

if train_type == "classifieronly":
    for param in model_env.get_classifieronly_params():
        param.requires_grad = True
else:
    for param in model_env.get_model().parameters():
        param.requires_grad = True

training_args = TrainingArguments(
    group,
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=30 if not is_test else 1,
    log_level="debug",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    learning_rate=5e-4 if "t5" in name else 5e-5,
    report_to=["wandb"],
    # push_to_hub=True,
    run_name=group
)

print(training_args.device)

model = model_env.get_model()

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset.get_train_split(),
    eval_dataset=dataset.get_eval_split(),
    data_collator=data_collator,
    tokenizer=model_env.get_tokenizer(),
    compute_metrics=compute_metrics,
)

trainer.train()


artifact = wandb.Artifact(name=f"model-{group}", type="model")

trainer.save_model(f"/model")
artifact.add_dir(f"/model")

run.log_artifact(artifact)