from datasets import load_dataset, interleave_datasets, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def load_winobias():
    result = {
        "validation": None,
        "test": None
    }
    for split in ["validation", "test"]:
        type1_pro = load_dataset("wino_bias", "type1_pro")[split]
        new_column = [1] * len(type1_pro)
        type1_pro = type1_pro.add_column("label", new_column)

        type2_pro = load_dataset("wino_bias", "type2_pro")[split]
        new_column = [1] * len(type2_pro)
        type2_pro = type2_pro.add_column("label", new_column)

        type1_anti = load_dataset("wino_bias", "type1_anti")[split]
        new_column = [0] * len(type1_anti)
        type1_anti = type1_anti.add_column("label", new_column)

        type2_anti = load_dataset("wino_bias", "type2_anti")[split]
        new_column = [0] * len(type2_anti)
        type2_anti = type2_anti.add_column("label", new_column)

        result[split] = interleave_datasets(
            [
                type1_pro,
                type1_anti,
                type2_pro,
                type2_anti
            ]
        )
    return DatasetDict({
        "train": result["validation"],
        "eval": result["test"]
    })


def process_winobias_split(dataset, tokenizer):
    def remove_tokenization(example):
        return {"sentence": " ".join(example["tokens"])}

    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True, padding=True)

    detokenized_dataset = dataset.map(remove_tokenization, batched=False)
    tokenized_dataset = detokenized_dataset.map(tokenize_function, batched=True, batch_size=32)
    tokenized_dataset = tokenized_dataset.remove_columns(['document_id', 'part_number', 'word_number', 'tokens', 'pos_tags', 'parse_bit', 'predicate_lemma', 'predicate_framenet_id', 'word_sense', 'speaker', 'ner_tags', 'verbal_predicates', 'coreference_clusters', 'sentence'])
    return tokenized_dataset


def process_winobias(tokenizer):
    data = load_winobias()
    return DatasetDict({
        "train": process_winobias_split(data["train"], tokenizer),
        "eval": process_winobias_split(data["eval"], tokenizer)
    })


def load_processed_winobias(tokenizer):
    dataset = load_dataset(f"henryscheible/winobias")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(dataset["eval"], batch_size=8, collate_fn=data_collator)
    return train_dataloader, eval_dataloader
