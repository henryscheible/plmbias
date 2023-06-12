from .base import StereotypeDataset
from .winobias import Winobias
from .crows_pairs import CrowsPairs
from .stereoset import Stereoset
from .implicit_bias import ImplicitBias

name_to_subclass = {
    "crows_pairs": CrowsPairs,
    "stereoset": Stereoset,
    "winobias": Winobias,
    "implicit_bias": ImplicitBias
}


def from_name(name, tokenizer, is_generative=False):
    return name_to_subclass[name](tokenizer, is_generative)


StereotypeDataset.from_name = from_name
