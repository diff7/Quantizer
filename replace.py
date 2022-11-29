import re
import copy
import logging
import torch
import torch.nn as nn
from typing import Dict, List
from dataclasses import asdict

from qconfig import QConfig
import layers as qlayers

MAPPING = {
    # nn.Linear: quant.Linear,
    nn.Conv2d: qlayers.SearchConv2d,
    # nn.Conv1d: quant.Conv1d,
}


def replace_modules(root_module, qconfig: QConfig, verbose: bool = False):

    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")

    new_dict = root_module._modules.copy()

    def apply(m):
        for name, child in m.named_children():
            verb(f"processing class {child} with name {name}")
            if type(child) in MAPPING:
                extended_class = MAPPING[type(child)]
                verb(f"replacing {child} with {extended_class}")
                initialized_module = extended_class.from_module(child, qconfig)
                setattr(m, name, initialized_module)
                new_dict[name] = initialized_module
            else:
                apply(child)

    apply(root_module)


def set_signle(root_module, verbose: bool = False):

    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")

    new_dict = root_module._modules.copy()

    def apply(m):
        for name, child in m.named_children():
            if hasattr(child, "set_single_conv"):
                verb(f"Setting fixed bit for {child}")
                initialized_module = getattr(child, "set_single_conv")()
                setattr(m, name, initialized_module)
                new_dict[name] = initialized_module
            else:
                apply(child)

    apply(root_module)


def get_alphas(root_module):
    alphas = []

    def apply(m):
        for name, child in m.named_children():
            if hasattr(child, "get_alphas"):
                alpha = getattr(child, "get_alphas")()
                alphas.append(alpha)
            else:
                apply(child)

    apply(root_module)

    return alphas
