import json
import os
import sys

import docker


def get_docker_contexts(contexts):
    return {context: docker.DockerClient(base_url=url) for context, url in contexts.items()}


def launch_experiments(experiments, context_urls):
    token = os.environ.get("HF_TOKEN")
    wandb_token = os.environ.get("WANBD_TOKEN")
    contexts = get_docker_contexts(context_urls)
    for experiment in experiments:
        client = contexts[experiment["context"]]
        if "buildargs" in experiment.keys():
            buildargs = {k: str(v) for k, v in experiment["buildargs"].items()}
            buildargs["GPU_CARD"] = str(experiment["card"])
            buildargs["TOKEN"] = token
            buildargs["WANDB_TOKEN"] = wandb_token
            print("Building image...")
            print(f"Image path: ./experiments/{experiment['image']}")
            image, _ = client.images.build(
                path=f"./experiments/{experiment['image']}",
                buildargs=buildargs,
                tag=experiment["name"]
            )
        else:
            buildargs = dict()
            buildargs["GPU_CARD"] = str(experiment["card"])
            buildargs["TOKEN"] = token
            print("Building image...")
            print(f"Image path: ./experiments/{experiment['image']}")
            image, _ = client.images.build(
                path=f"./experiments/{experiment['image']}",
                buildargs=buildargs,
                tag=experiment["name"]
            )
        print("Launching container...")
        os.system(f"docker context use {experiment['context']} && docker run -e WANDB_DOCKER={experiment['name']} -itd --gpus all --name {experiment['name']} {experiment['name']}")
        print(f"Started Experiment: {experiment['name']}")


def build_images(images, context_urls):
    token = os.environ.get("HF_TOKEN")
    wandb_token = os.environ.get("WANBD_TOKEN")
    contexts = get_docker_contexts(context_urls)
    for image in images:
        for key, client in contexts.items():
            print(f"Building image {image['image']} in context {key}")
            _, _ = client.images.build(
                path=f"./experiments/{image['image']}",
                buildargs={"TOKEN": token},
                tag=image["image"]
            )


def monitor_experiments(experiments, context_urls):
    contexts = get_docker_contexts(context_urls)
    print(f"\033[94m \033[1m{'Name':<50} \033[0m{'Machine':<12}  {'Card':<5} {'Status':<10} | {'Logs'}")
    for experiment in experiments:
        client = contexts[experiment["context"]]
        try:
            container = client.containers.get(experiment["name"])
            print(f"\033[94m \033[1m{experiment['name']:<50} \033[0m{experiment['context']:<12}  {experiment['card']:<5} {container.status:<10} | {str(container.logs(tail=1))[:100]}")
        except docker.errors.NotFound:
            print(f"Container \"{experiment['name']}\" does not exist")


def stop_experiments(experiments, context_urls):
    contexts = get_docker_contexts(context_urls)
    for experiment in experiments:
        client = contexts[experiment["context"]]
        try:
            container = client.containers.get(experiment["name"])
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            print(f"Container \"{experiment['name']}\" does not exist")


if __name__ == "__main__":
    with open(sys.argv[2]) as file:
        argStr = "".join(file.readlines())
    obj = json.loads(argStr)
    prebuild = obj["prebuild"] if "prebuild" in obj.keys() else None
    experiments = obj["experiments"]
    contexts = obj["contexts"]
    if sys.argv[1] == "launch":
        if prebuild is not None:
            build_images(prebuild, contexts)
        launch_experiments(experiments, contexts)
    elif sys.argv[1] == "monitor":
        monitor_experiments(experiments, contexts)
    elif sys.argv[1] == "stop":
        stop_experiments(experiments, contexts)
    else:
        print("Invalid Command")
