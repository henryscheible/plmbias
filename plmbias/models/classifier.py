import numpy as np
import torch
from torch import nn

from plmbias.models.base import ModelEnvironment
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_to_params = {
    "bert-base-uncased": [
        "classifier.weight",
        "classifier.bias"
    ],
    "bert-large-uncased": [
        "classifier.weight",
        "classifier.bias"
    ],
    "roberta-base": [
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    ],
    "roberta-large": [
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    ],
    "xlnet-base-cased": [
        "sequence_summary.summary.weight"
        "sequence_summary.summary.bias",
        "logits_proj.weight",
        "logits_proj.bias"
    ],
    "xlnet-large-cased": [
        "sequence_summary.summary.weight"
        "sequence_summary.summary.bias",
        "logits_proj.weight",
        "logits_proj.bias"
    ],
    "gpt2": [
        "score.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias"
    ]

}


class SequenceClassificationEnvironment(ModelEnvironment):
    def __init__(self, hf_model_id: str):
        super().__init__()
        self.model_id = hf_model_id
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if "gpt" in hf_model_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        #     self.model.score = nn.Linear(self.model.score.in_features, self.model.score.out_features, bias=True)
        #     self.model.post_init()

    def get_compute_metrics_fn(self):
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

        return compute_metrics

    def evaluate_batch(self, eval_batch, mask, metric):
        shape = self.get_mask_shape()
        with torch.no_grad():
            outputs = self.model(**eval_batch, head_mask=mask.reshape(shape))
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=eval_batch["labels"])

    def get_classifieronly_params(self):
        return model_to_params[self.model_id]

    def setup_dataset(self, dataset):
        dataset.process()
