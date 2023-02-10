from plmbias.models.base import ModelEnvironment
from transformers import BertForSequenceClassification


class CausalLMEnvironment(ModelEnvironment):
    def __init__(self, hf_model_id: str):
        self.model = BertForSequenceClassification.from_pretrained(hf_model_id)



