import os

os.environ["MODEL"] = "t5-small"
os.environ["TRAIN_TYPE"] = "classifieronly"
os.environ["DATASET"] = "winobias"
os.environ["MODEL_TYPE"] = "generative"

import train




