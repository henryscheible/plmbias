import numpy as np
import torch
import evaluate

from typing import Dict, Mapping, Optional, Any, Union, List, Tuple

from torch import nn

from plmbias.datasets import StereotypeDataset
from plmbias.models.base import ModelEnvironment
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, Trainer

max_source_length = 512
max_target_length = 128

def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach()

class GenerativeEnvironment(ModelEnvironment):
    def __init__(self, hf_model_id: str):
        super().__init__()
        self.model_id = hf_model_id
        self.model = T5ForConditionalGeneration.from_pretrained(hf_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        self.acc = evaluate.load("accuracy")

    def get_classifieronly_params(self):
        return self.model.decoder.parameters()

    def get_compute_metrics_fn(self):
        true_label_id = self.tokenizer("true").input_ids[0]
        false_label_id = self.tokenizer("false").input_ids[0]

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            true_logit = logits[0][:, 0, true_label_id]
            false_logit = logits[0][:, 0, false_label_id]
            binary_logits = np.stack([false_logit, true_logit], axis=-1)
            predictions = np.argmax(binary_logits, axis=-1)
            print(predictions.shape)
            binary_labels = np.array(list(map(lambda label: 0 if label[0] == false_label_id else 1, labels)))
            print(binary_labels.shape)
            print(list(zip(predictions, binary_labels)))
            confusion_matrix = np.zeros((2, 2))
            for label, pred in zip(binary_labels, predictions):
                confusion_matrix[label, pred] += 1
            return {
                "accuracy": np.sum(predictions == binary_labels) / float(len(binary_labels)),
                "tp": confusion_matrix[1, 1] / float(len(binary_labels)),
                "tn": confusion_matrix[0, 0] / float(len(binary_labels)),
                "fp": confusion_matrix[0, 1] / float(len(binary_labels)),
                "fn": confusion_matrix[1, 0] / float(len(binary_labels)),
            }

        return compute_metrics

    def setup_dataset(self, dataset: StereotypeDataset):
        dataset.process()
        train_split = dataset.get_train_split()
        eval_split = dataset.get_eval_split()

        def convert_labels(example):
            return {"labels": "true"} if example["label"] == 1 else {"labels": "false"}

        def tokenize_fn(example):
            target_encoding = self.tokenizer(
                example["labels"],
                padding=True,
                max_length=max_target_length,
                truncation=True,
                return_tensors="pt",
            )
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"labels": labels}

        def tokenize_fn_eval(example):
            target_encoding = self.tokenizer(
                example["labels"],
                padding=True,
                max_length=max_target_length,
                truncation=True,
                return_tensors="pt",
            )
            labels = target_encoding.input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"labels": labels}

        train_split = train_split.map(convert_labels, batched=False, remove_columns="label")
        train_split = train_split.map(tokenize_fn, batched=True, batch_size=32)

        eval_split = eval_split.map(convert_labels, batched=False, remove_columns="label")
        eval_split = eval_split.map(tokenize_fn_eval, batched=True, batch_size=32)

        dataset.train_split = train_split
        dataset.eval_split = eval_split

    def evaluate_batch(self, eval_batch, metric, encoder_mask = None, decoder_mask = None):
        shape = self.get_mask_shape()
        decoder_shape = self.get_mask_shape_decoder()

        batch_size = len(eval_batch["input_ids"])
        pad_token_id = self.model.config.pad_token_id
        decoder_input_ids = torch.tensor([[pad_token_id]] * batch_size).to(self.model.device)
        eval_batch["decoder_input_ids"] = decoder_input_ids
        labels = eval_batch.pop("labels")
        
        with torch.no_grad():
            outputs = self.model(**eval_batch, head_mask=encoder_mask.reshape(shape) if encoder_mask is not None else None,
                                 decoder_head_mask=decoder_mask.reshape(decoder_shape) if decoder_mask is not None else None)
        logits = outputs.logits
        true_label_id = self.tokenizer("true").input_ids[0]
        false_label_id = self.tokenizer("false").input_ids[0]
        true_logit = logits[:, 0, true_label_id].detach().cpu().numpy()
        false_logit = logits[:, 0, false_label_id].detach().cpu().numpy()
        binary_logits = np.stack([false_logit, true_logit], axis=-1)
        predictions = np.argmax(binary_logits, axis=-1)
        binary_labels = np.array(list(map(lambda label: 0 if label[0] == false_label_id else 1, labels)))
        metric.add_batch(predictions=predictions, references=binary_labels)

    def get_mask_shape(self):
        return torch.Size([
            self.model.config.num_hidden_layers,
            self.model.config.num_attention_heads
        ])
    
    def get_mask_shape_decoder(self):
        return torch.Size([
            self.model.config.num_decoder_layers,
            self.model.config.num_attention_heads
        ])