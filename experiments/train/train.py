import os
import datetime

import evaluate
import numpy as np
import torch
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments

from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment

# os.environ["MODEL"] = "gpt2"
# os.environ["DATASET"] = "crows_pairs"
# os.environ["TRAIN_TYPE"] = "finetuned"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "true"

hf_model_id = os.environ.get("MODEL")
train_type = os.environ.get("TRAIN_TYPE")
dataset = os.environ.get("DATASET")
lr = float(os.environ.get("LR"))

name = f"{hf_model_id}_{dataset}_{train_type}"

model_env = ModelEnvironment.from_pretrained(hf_model_id)

dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
data_collator = DataCollatorWithPadding(model_env.get_tokenizer())


os.environ["WANDB_PROJECT"] = name


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    confusion_matrix = np.zeros((2, 2))
    for label, pred in zip(labels, predictions):
        confusion_matrix[label, pred] += 1
    return {
        "accuracy": np.sum(predictions == labels) / float(len(labels)),
        "tp": confusion_matrix[1, 1] / float(len(labels)),
        "tn": confusion_matrix[0, 0] / float(len(labels)),
        "fp": confusion_matrix[0, 1] / float(len(labels)),
        "fn": confusion_matrix[1, 0] / float(len(labels)),
    }


if train_type == "classifieronly":
    for param in model_env.get_model().base_model.parameters():
        param.requires_grad = False
else:
    for param in model_env.get_model().parameters():
        param.requires_grad = True

training_args = TrainingArguments(
    name,
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=80,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=50,
    log_level="debug",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    push_to_hub=True,
    learning_rate=lr,
    report_to=["wandb"],
    run_name=f"{lr:e}"
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
print("pushing to hub")
trainer.push_to_hub()
print("done pushing to hub")
