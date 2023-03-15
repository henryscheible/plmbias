import pandas as pd
from tqdm import tqdm
import requests
import sys
import json

for launch_file in sys.argv[1:]:
    with open(launch_file, "r") as file:
        data = json.loads("".join(file.readlines()))
    results = pd.DataFrame(columns=["name", "accuracy", "has_contribs"])
    results["name"] = [experiment["name"] for experiment in data["experiments"]]
    results = results.set_index("name")
    for experiment in tqdm(data["experiments"]):
        try:
            validation = requests.get(f"https://huggingface.co/henryscheible/{experiment['name']}/raw/main/README.md").text
            val_lines = validation.split("\n")
            acc_line = list(filter(lambda l: "Accuracy:" in l, val_lines))[0]
            acc = acc_line[12:]
            results.loc[experiment["name"]]["accuracy"] = acc
        except:
            pass
        try:
            contribs = requests.get(
                f"https://huggingface.co/henryscheible/{experiment['name']}/raw/main/contribs.txt").text
            if contribs[0] == "[":
                results.loc[experiment["name"]]["has_contribs"] = True
        except:
            pass
    print(results)


