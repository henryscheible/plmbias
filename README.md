# PLMBias

## Structure
### `/plmbias`
This directory is a python package containing code shared across multiple experiments. It is used as the base upon which the other experiments are built.

Important Classes/functions (not an exhaustive list):
* `plmbias.models.ModelEnvironment.from_pretrained(hf_model_id: str)`
  * Returns a sequence classification model environment (`ModelEnvironment`) pulled from huggingface (using hf_model_id.) Note that this can a model provided by huggingface (ex: "gpt2"), or a model finetuned using this library (ex: "henryscheible/gpt2_stereoset_finetuned"). 
* `plmbias.models.ModelEnvironment.from_pretrained_lm(hf_model_id: str)`
  * Returns a Causal language model environment (`ModelEnvironment`) pulled from huggingface (using hf_model_id.) Note that this can a model provided by huggingface (ex: "gpt2"), or a model finetuned using this library (ex: "henryscheible/gpt2_stereoset_finetuned"). 
  * Note that this model does not have to be trained on Causal LM, classifier weights will be ignored if necessary
* `plmbias.models.ModelEnvironment`:
  * This class should not be directly instantiated, use one of the two methods above.
  * `get_model()`
    * Returns the model object (of type `PreTrainedModel`)
  * `get_tokenizer()`
    * Returns the tokenizer object
  * `get_mask_shape()`
    * Returns the size of the head mask appropriate for this model: (num_hidden_layers, num_attention_heads) (returns a `torch.Size` object)
* `plmbias.datasets.StereotypeDataset.from_name(dataset_name, tokenizer)`
  * Instantiates a `StereotypeDataset` from a given dataset name and tokenizer object
  * `dataset_name` must be one of: `"crows_pairs"`, `"stereoset"`, `"winobias"`
* `plmbias.datasets.StereotypeDataset`
  * This class should not be directly instantiated, use the method above
  * `get_train_split()`
    * Returns the training split of the dataset
  * `get_eval_split()`
    * Returns the evaluation split of the dataset

#### `/Dockerfile`
This dockerfile builds the basic `plmbias` docker image. This image
* Installs all required dependenceis, including pytorch, huggingface, scikit-learn, ect
* Copies in the plmbias python package

### `/experiments`
The `/experiments` directory contains build contexts for docker images. Each subdirectory is organized as follows:
* `/<experiment name>`
  * `Dockerfile`: Defines the build method for the docker image.
    * Inherits from the main `plmbias` docker image
    * Takes build arguments and sets them up as environment variables inside the docker container
    * Copies `train.py` into the docker image and sets the container entrypoint to running that script
  * `train.py`: Training script. This is what is actually run inside the docker container. 
    * Specifications are taken from environment variables (see above)
    * Scripts generally end with all models/results/etc being pushed to a huggingface repository.

Example Dockerfile with added comments/annotations: `/experiments/train/Dockerfile`
```dockerfile
# Starts building from the main plmbias Docker image
FROM plmbias
# Takes a HuggingFace token as a build argument
ARG TOKEN
# Authenticates with huggingface, then checks this authentication. The build will fail here if the token is invalid
RUN python3 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$TOKEN')"
RUN python -c 'from huggingface_hub import whoami; print(whoami())'
# Takes model to finetune from, training_type, dataset to train on, gpu card, and learning rate as arguments
ARG MODEL
ARG TRAIN_TYPE
ARG DATASET
ARG GPU_CARD
ARG LR
# Saves these arguments as environment variables so they can be read by the python script
ENV MODEL=$MODEL
ENV TRAIN_TYPE=$TRAIN_TYPE
ENV DATASET=$DATASET
ENV LR=$LR
# These arguments are also saved as environment variables, but will be read directly by pytorch, cuda, and huggingface tranformers rather than user code
ENV CUDA_VISIBLE_DEVICES=$GPU_CARD
ENV TOKENIZERS_PARALLELISM=false
# Copies train.py into the image
COPY ./train.py /workspace
# Sets train.py as the entrypoint of the resulting container 
CMD ["python3", "/workspace/train.py"]
```

### `/groups`
This folder contains JSON launch files and python scripts for automatically generating those launch files. A launch file contains a specification of which docker images from `/experiments` to build and run on which machines, which gpu cards, and which build arguments to pass. These JSON files are designed to work with the `run_experiments.py` script. See the section below for usage details.

## Setup
To use this repository, you will need to change the following things:
* Setup SSH keys for any remote server you wish to run an experiment on (password authentication will not work)
* Create a docker context for each server
  * Ex to create a context named mms-large-1: `$ docker context create --docker host=ssh://henry@mms-large-1.cs.dartmouth.edu mms-large-1`. NOTE: CREATE THIS CONTEXT ON YOUR MACHINE, NOT ON THE SERVER. The context allows your computer to connect to the server to run docker commands without explicitly sshing, i.e. running `docker ps` on your laptop (after selecting the context as described below) will result in a list of containers on the server, not on your laptop. Docker will handle creating an ssh connection in the background for you. If you prefer not to use docker contexts, they are not required but then you have to explicity ssh in to the server to run commands. 
* Edit the "contexts" field in each JSON file in groups to contain the correct hostnames and context names
* Create a HuggingFace access token and add it to the `HF_TOKEN` environment variable

## Usage
Running experiments centers around the `run_experiment.py`
* Build the plmbias image on each server
  * Select the context with `$ docker context use <context_name>`
  * Build the image with `$ docker build . -t plmbias`
* Launch the appropriate configuration file with
  * `$ python3 run_experiment.py launch groups/<group_name>.json`
* Monitor progress with:
  * `$ python3 run_experiment.py monitor groups/<group_name>.json`
* Stop a group and clean up resources with:
  * `$ python3 run_experiment.py stop groups/<group_name>.json`
  * Note that this can take a weirdly long time (1-2 minutes), this is normal
