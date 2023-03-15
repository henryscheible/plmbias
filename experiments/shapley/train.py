import os

import evaluate
import transformers
import torch
from captum.attr import ShapleyValueSampling
import json
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from plmbias.datasets import StereotypeDataset
from plmbias.models import ModelEnvironment


def attribute_factory(model, eval_dataloader, shape):
    def attribute(mask):
        mask = mask.flatten()
        metric = evaluate.load("accuracy")
        model.eval()
        for eval_batch in eval_dataloader:
            eval_batch = {k: v.to("cuda") for k, v in eval_batch.items()}
            with torch.no_grad():
                outputs = model(**eval_batch, head_mask=mask.reshape(shape))
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=eval_batch["labels"])
        return metric.compute()["accuracy"]

    return attribute

def get_shapley(eval_dataloader, model_env, num_samples=250, num_perturbations_per_eval=1):
    transformers.logging.set_verbosity_error()

    mask = torch.ones(model_env.get_mask_shape()).to("cuda").flatten().unsqueeze(0)
    model = model_env.get_model().to("cuda")
    attribute = attribute_factory(model, eval_dataloader, model_env.get_mask_shape())

    with torch.no_grad():
        model.eval()
        sv = ShapleyValueSampling(attribute)
        attribution = sv.attribute(
            mask, n_samples=num_samples, show_progress=True,
            perturbations_per_eval=num_perturbations_per_eval
        )

    print(attribution)

    with open("contribs.txt", "w") as file:
        file.write(json.dumps(attribution.flatten().tolist()))

    return attribution


checkpoint = os.environ.get("CHECKPOINT")
dataset = os.environ.get("DATASET")

model_env = ModelEnvironment.from_pretrained(checkpoint)

dataset = StereotypeDataset.from_name(dataset, model_env.get_tokenizer())
data_collator = DataCollatorWithPadding(model_env.get_tokenizer())
eval_dataloader = DataLoader(dataset.get_eval_split(), shuffle=True, batch_size=1024, collate_fn=data_collator)
print("")
get_shapley(eval_dataloader, model_env, num_samples=250)

api = HfApi()
api.upload_file(
    path_or_fileobj="contribs.txt",
    path_in_repo="contribs.txt",
    repo_id=checkpoint,
    repo_type="model"
)
