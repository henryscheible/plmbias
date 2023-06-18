from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch

model_dir = "<MODEL DIRECTORY>"

model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

input_sentence = "<PUT SENTENCE HERE>"

inputs = tokenizer(input_sentence, padding=True, return_tensors="pt")

outputs = model(**inputs)

logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)
print(prediction)
