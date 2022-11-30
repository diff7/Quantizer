import torch
import torch.nn as nn

from QSB.qconfig import QConfig
import QSB.layers as qlayers

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


# Not nice but need some flag to filter alphas and other params
def get_alphas_and_set_grad_true(root_module):
    alphas = []
    names = []

    def apply(m):
        for name, child in m.named_children():
            if hasattr(child, "get_alphas"):
                alpha = getattr(child, "get_alphas")()
                alpha.requires_grad = True
                alphas.append(alpha)
                names.append(name)
            else:
                apply(child)

    apply(root_module)

    return alphas, names


def get_flops_and_memory(root_module, input_size: list = (1, 3, 28, 28)):

    input_x = torch.randn(*input_size)
    root_module(input_x)

    flops = 0
    memory = 0

    def apply(module, fl, mem):
        for name, child in module.named_children():
            if hasattr(child, "fetch_info"):
                f, m = getattr(child, "fetch_info")()
                fl += f
                mem += m
            else:
                apply(child, fl, mem)

        return fl, mem

    return apply(root_module, flops, memory)


## Mutable approach ##

# def get_flops_and_memory_(root_module):

#     info = [0, 0]

#     def apply(module):
#         for name, child in module.named_children():
#             if hasattr(child, "fetch_info"):
#                 f, m = getattr(child, "fetch_info")()
#                 info[0] = info[0] + f
#                 info[1] = info[1] + m
#             else:
#                 apply(child)

#     apply(root_module)
#     return tuple(info)


def prepare_and_get_params(model, qconfig, verbose=True):
    replace_modules(model, qconfig, verbose=verbose)
    main_parms = [p for p in model.parameters() if p.requires_grad]
    alphas, names = get_alphas_and_set_grad_true(model)
    return model, main_parms, alphas
