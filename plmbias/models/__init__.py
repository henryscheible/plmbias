from .base import ModelEnvironment
from .classifier import SequenceClassificationEnvironment

model_classes = {
    "bert-base-uncased": SequenceClassificationEnvironment,
    "bert-large-uncased": SequenceClassificationEnvironment,
    "roberta-base": SequenceClassificationEnvironment,
    "roberta-large": SequenceClassificationEnvironment,
    "xlnet-base-cased": SequenceClassificationEnvironment,
    "xlnet-large-cased": SequenceClassificationEnvironment,
    "gpt2": SequenceClassificationEnvironment,
    "gpt2-medium": SequenceClassificationEnvironment,
    "gpt2-large": SequenceClassificationEnvironment
}


def from_pretrained(hf_model_id: str):
    return model_classes[hf_model_id](hf_model_id)


ModelEnvironment.from_pretrained = from_pretrained