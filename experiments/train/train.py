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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

is_test = os.environ.get("IS_TEST") == "true"
 
os.environ["WANDB_MODE"] = "online"

if is_test:
    os.environ["MODEL"] = "t5-small"
    os.environ["DATASET"] = "implicit_bias"
    os.environ["TRAIN_TYPE"] = "finetuned"
    os.environ["LEARNING_RATE"] = "5e-5"
    os.environ["MODEL_TYPE"] = "generative"

config = dict()

rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

seed = int(os.environ.get("SEED")) if os.environ.get("SEED") is not None else 42

transformers.set_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
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
run = wandb.init(name=group, project="plmbias" if not is_test else "plmbias-test", group="train")
num_train_epochs = int(os.environ.get("EPOCHS")) if os.environ.get("EPOCHS") is not None else 20
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

data_collator = DataCollatorWithPadding(model_env.get_tokenizer(), padding=True, max_length=100)

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
    eval_steps=20 if is_test else 20,
    save_strategy="steps",
    save_steps=20,
    max_steps=-1 if is_test else -1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=num_train_epochs if not is_test else 20,
    log_level="debug",
    load_best_model_at_end=True,
    save_total_limit=1,
    metric_for_best_model="accuracy",
    logging_steps=10,
    learning_rate=learning_rate,
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