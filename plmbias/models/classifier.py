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

    def get_classifieronly_params(self):
        return model_to_params[self.model_id]
