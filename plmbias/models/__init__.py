from .base import ModelEnvironment
from .causal import CausalLMEnvironment
from .classifier import SequenceClassificationEnvironment
from .generative import GenerativeEnvironment

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
    return SequenceClassificationEnvironment(hf_model_id)

def from_pretrained_lm(hf_model_id: str):
    return CausalLMEnvironment(hf_model_id)

def from_pretrained_generative(hf_model_id: str):
    return GenerativeEnvironment(hf_model_id)


ModelEnvironment.from_pretrained = from_pretrained
ModelEnvironment.from_pretrained_lm = from_pretrained_lm
ModelEnvironment.from_pretrained_generative = from_pretrained_generative
