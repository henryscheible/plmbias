from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np
import evaluate


def load_crows_pairs():
    return load_dataset("crows_pairs")['test']


def process_crows_pairs(dataset, tokenizer, batch_size=64):
    def add_label(example):
        if example["label"] == 1:
            return {
                "sent_a": example["sent_more"],
                "sent_b": example["sent_less"]
            }
        else:
            return {
                "sent_a": example["sent_less"],
                "sent_b": example["sent_more"]
            }

    def tokenize(example):
        return tokenizer(example["sent_a"], example["sent_b"], truncation=True, padding=True)

    num_samples = len(dataset["sent_more"])
    dataset = dataset.remove_columns([
        "stereo_antistereo",
        "bias_type",
        "annotations",
        "anon_writer",
        "anon_annotators",
    ])
    dataset = dataset.add_column("label", np.random.choice(2, num_samples))
    dataset_processed = dataset.map(add_label, batched=False)
    tokenized_dataset = dataset_processed.map(tokenize, batched=True, batch_size=64)
    tokenized_dataset = tokenized_dataset.remove_columns([
        "id",
        "sent_more",
        "sent_less",
        "sent_a",
        "sent_b"
    ])
    split_tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=0.2
    )
    return split_tokenized_dataset


def load_processed_crows_pairs(tokenizer):
    dataset = load_dataset(f"henryscheible/crows_pairs")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(dataset["test"], batch_size=8, collate_fn=data_collator)
    return train_dataloader, eval_dataloader


