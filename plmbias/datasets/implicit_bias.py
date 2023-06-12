from datasets import load_dataset
from plmbias.datasets import StereotypeDataset
from retry.api import retry_call

class ImplicitBias(StereotypeDataset):

    def process(self):
        dataset = retry_call(load_dataset, fargs=["henryscheible/implicit_bias"], tries=20, delay=2)

        def tokenize(example):
            return self.tokenizer(example["sentence"], truncation=True, padding=True)

        dataset = dataset.remove_columns([
            "category"
        ])
        self.train_split = dataset["train"].map(tokenize, batched=True, batch_size=64,
                                                  remove_columns=["sentence"])
        self.eval_split = dataset["test"].map(tokenize, batched=True, batch_size=64,
                                                  remove_columns=["sentence"])


