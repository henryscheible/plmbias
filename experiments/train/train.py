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

config = dict()
# config["MODEL"] = "t5-small"
# config["DATASET"] = "stereoset"
# config["TRAIN_TYPE"] = "finetuned"
# config["per_device_train_batch_size"] = 64
# config["learning_rate"] = 5e-4
# config["adam_beta1"] = 0.9
# config["adam_beta2"] = 0.999
# config["adam_epsilon"] = 1e-8


os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_WATCH"] = "all"


rand_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))


transformers.set_seed(42)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)
transformers.enable_full_determinism(42)

config = wandb.config
hf_model_id = os.environ.get("MODEL")
train_type = os.environ.get("TRAIN_TYPE")
dataset = os.environ.get("DATASET")
model_type = os.environ.get("MODEL_TYPE")
group = f"{hf_model_id.replace('/', '-')}_{dataset}_{train_type}"
name = f"{group}_{rand_id}"
run = wandb.init(name=name, project="plmbias")
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


os.environ["WANDB_PROJECT"] = name

compute_metrics = model_env.get_compute_metrics_fn()

for param in model_env.get_classifieronly_params():
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
    num_train_epochs=30,
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

trainer = Trainer(
    model_env.get_model(),
    training_args,
    train_dataset=dataset.get_train_split(),
    eval_dataset=dataset.get_eval_split(),
    data_collator=data_collator,
    tokenizer=model_env.get_tokenizer(),
    compute_metrics=compute_metrics,
)

trainer.train()
