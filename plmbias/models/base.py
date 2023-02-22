import torch

class ModelEnvironment:

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_mask_shape(self):
        return torch.Size([
            self.model.config.num_hidden_layers,
            self.model.config.num_attention_heads
        ])