from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from plmbias.datasets import StereotypeDataset


class Stereoset(StereotypeDataset):

    def process(self):

        def process_fn(example):
            sentences = []
            labels = []
            contexts = []
            for i in range(3):
                print(f"example: {example}")
                if example["sentences"][0]["gold_label"][i] != 2:
                    sentences.append(example["sentences"][0]["sentence"][i])
                    labels.append(example["sentences"][0]["gold_label"][i])
                    contexts.append(example["context"][0])
            return {
                "sentence": sentences,
                "label": labels,
                "context": contexts
            }

        dataset = load_dataset("stereoset", "intersentence")['validation']

        def tokenize(example):
            return self.tokenizer(example["context"], example["sentence"], truncation=True, padding=True)

        num_samples = len(dataset["id"])
        dataset = dataset.remove_columns([
            "id",
            "target",
            "bias_type",
        ])
        dataset_processed = dataset.map(process_fn, batched=True, batch_size=1, remove_columns=["sentences"])
        print(dataset_processed.column_names)
        tokenized_dataset = dataset_processed.map(tokenize, batched=True, batch_size=64,
                                                  remove_columns=["context", "sentence"])
        print(tokenized_dataset.column_names)

        split_tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=0.3
        )

        self.train_split = split_tokenized_dataset["train"]
        self.eval_split = split_tokenized_dataset["test"]

