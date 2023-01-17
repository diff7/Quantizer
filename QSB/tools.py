import torch
import torch.nn as nn
from QSB.qconfig import QConfig
import QSB.layers as qlayers


# TODO
# Set some arbitrary arch
# Get an arbitrary arch

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
    # new_dict = root_module._modules.copy()

    searchable_params = []
    params_names = []

    def apply(m, current_name):
        for name, child in m.named_children():
            # verb(f"processing class {child} with name {name}")
            if type(child) in MAPPING:
                extended_class = MAPPING[type(child)]
                verb(f"replacing {name} {child} with {extended_class}")
                initialized_module = extended_class.from_module(child, qconfig)
                setattr(m, name, initialized_module)
                searchable_params.append(initialized_module.get_alphas())

                # TODO change 'conv_func' for different function names later

                params_names.append(
                    f"{current_name}{'.' if len(current_name)>0  else ''}{name}.conv_func.alphas"
                )
            else:
                store_name = current_name
                current_name = (
                    f"{current_name}{'.' if len(current_name)>0  else ''}{name}"
                )
                apply(child, current_name=current_name)
                current_name = store_name

    apply(root_module, current_name="")
    return searchable_params, params_names


def set_best_single(root_module, verbose: bool = False):

    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")

    # new_dict = root_module._modules.copy()

    def apply(m):
        for name, child in m.named_children():
            if hasattr(child, "set_single_conv"):
                verb(f"Setting fixed bit for {child}")
                initialized_module = getattr(child, "set_single_conv")(
                    use_max=True
                )
                setattr(m, name, initialized_module)
                # new_dict[name] = initialized_module
            else:
                apply(child)

    apply(root_module)
    return root_module


def get_named_arch(root_module, verbose: bool = False):
    """Returns a dict with layer names and best bit values"""

    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")
    # new_dict = root_module._modules.copy()

    arch = dict()
    arch_vector = dict()

    def apply(m, current_name):
        for name, child in m.named_children():
            # verb(f"processing class {child} with name {name}")
            if hasattr(child, "get_arch_values"):
                bit, alphas = getattr(child, "get_arch_values")()
                op_name = f"{current_name}{'.' if len(current_name)>0  else ''}{name}.conv_func.alphas"
                print(name)
                print(op_name)
                arch[op_name] = bit
                arch_vector[op_name] = alphas
            else:
                store_name = current_name
                current_name = (
                    f"{current_name}{'.' if len(current_name)>0  else ''}{name}"
                )
                apply(child, current_name=current_name)
                current_name = store_name

    apply(root_module, current_name="")
    return arch, arch_vector


def set_named_arch(root_module, arch, verbose: bool = False):

    if verbose:
        verb = print
    else:

        def verb(*args, **kwargs):
            pass

    verb("\n###############################")
    # new_dict = root_module._modules.copy()

    def apply(m, current_name):
        for name, child in m.named_children():
            # verb(f"processing class {child} with name {name}")
            if hasattr(child, "set_single_conv"):
                op_name = f"{current_name}{'.' if len(current_name)>0  else ''}{name}.conv_func.alphas"                
                if op_name in arch:
                    bit = arch[op_name]
                    initialized_module = getattr(child, "set_single_conv")(
                        bit=bit
                    )
                    setattr(m, name, initialized_module)
            else:
                store_name = current_name
                current_name = (
                    f"{current_name}{'.' if len(current_name)>0  else ''}{name}"
                )
                apply(child, current_name=current_name)
                current_name = store_name

    apply(root_module, current_name="")
    return root_module


def get_flops_and_memory(
    root_module, use_cached=False, input_size=None, device="cpu"
):

    if not use_cached:
        assert (
            not input_size is None
        ), "Provide input_size to create a dummy tensor for FLOPS computation or use_cached=True"

        input_x = torch.randn(*input_size).to(device)
        root_module(input_x)

    mem = []
    fl = []

    def apply(module):
        for name, child in module.named_children():
            if hasattr(child, "fetch_info"):
                f, m = getattr(child, "fetch_info")()
                fl.append(f)
                mem.append(m)
            else:
                apply(child)
        return fl, mem

    f, m = apply(root_module)
    return sum(f), sum(m)


def prepare_and_get_params(model, qconfig, verbose=True):
    alphas, alpha_names = replace_modules(model, qconfig, verbose=verbose)
    main_parms = [
        p for n, p in model.named_parameters() if not n in alpha_names
    ]

    return model, main_parms, alphas, alpha_names


# # Not nice but need some flag to filter alphas and other params
# def get_alphas_and_set_grad_true(root_module, grad=True):
#     alphas = []
#     names = []

#     def apply(m):
#         for name, child in m.named_children():
#             if hasattr(child, "get_alphas"):
#                 alpha = getattr(child, "get_alphas")()
#                 if grad:
#                     alpha.requires_grad = True
#                 alphas.append(alpha)
#                 names.append(name)

#             else:
#                 apply(child)

#     apply(root_module)
#     return alphas, names


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
# def get_alphas(root_module):
#     return get_alphas_and_set_grad_true(root_module, grad=False)
