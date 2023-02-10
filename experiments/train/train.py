import os

import evaluate
import numpy as np
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments

from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment

hf_model_id = os.environ.get("MODEL")
train_type = os.environ.get("TRAIN_TYPE")
dataset = os.environ.get("DATASET")

name = f"{hf_model_id}_{dataset}_{train_type}"

model_env = ModelEnvironment.from_pretrained(hf_model_id)

dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
data_collator = DataCollatorWithPadding(model_env.get_tokenizer())

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if train_type == "classifieronly":
    for param in model_env.get_model().base_model.parameters():
        param.requires_grad = False

training_args = TrainingArguments(
    name,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=6,
    log_level="info",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

print(training_args.device)
#
# trainer = Trainer(
#     model_env.get_model(),
#     training_args,
#     train_dataset=dataset.get_train_split(),
#     eval_dataset=dataset.get_eval_split(),
#     data_collator=data_collator,
#     tokenizer=model_env.get_tokenizer(),
#     compute_metrics=compute_metrics
# )

# trainer.train()
