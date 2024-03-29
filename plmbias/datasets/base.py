
class StereotypeDataset:
    def __init__(self, tokenizer, is_generative=False):
        self.eval_split = None
        self.train_split = None
        self.tokenizer = tokenizer
        self.is_generative = is_generative

    def get_train_split(self):
        if self.train_split is None:
            self.process()
        return self.train_split

    def get_eval_split(self):
        if self.eval_split is None:
            self.process()
        return self.eval_split

    def process(self):
        pass
