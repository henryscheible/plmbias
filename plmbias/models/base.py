
class ModelEnvironment:

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
