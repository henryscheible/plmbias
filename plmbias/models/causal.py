from plmbias.models.base import ModelEnvironment
from transformers import AutoTokenizer, AutoModelForCausalLM

from plmbias.models.classifier import model_to_params


class CausalLMEnvironment(ModelEnvironment):
    def __init__(self, hf_model_id: str):
        super().__init__()
        self.model_id = hf_model_id
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_id)
        if "gpt2" in hf_model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, mask_token="<|endoftext|>")
        if "bert" in hf_model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if "xlnet" in hf_model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if "gpt" in hf_model_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        #     self.model.score = nn.Linear(self.model.score.in_features, self.model.score.out_features, bias=True)
        #     self.model.post_init()

    def get_classifieronly_params(self):
        return model_to_params[self.model_id]