import torch

from QSB.qconfig import QConfig
from QSB.tools import (
    replace_modules,
    set_best_single,
    get_flops_and_memory,
    prepare_and_get_params,
    get_named_arch,
    set_named_arch,
)

from QSB.quantizers import HWGQ, LsqQuan

from models.simple import SimpleCNN
from models.IMDN.architechture import IMDN

weight = lambda bit, weight: LsqQuan(
    bit, all_positive=False, symmetric=False, per_channel=False, weight=weight
)

act = lambda bit, weight: HWGQ(bit)

qconfig = QConfig(weight_quantizer=weight, act_quantizer=act)
model = SimpleCNN()
# model = IMDN()

input_x = torch.randn(10, 3, 28, 28)
model(input_x)

model, main_params, alpha, alpha_names = prepare_and_get_params(
    model, qconfig, verbose=True
)
print("#" * 10)
print("\n \n", alpha_names)
print(len(alpha_names))

arch, arch_vector = get_named_arch(model)
print(arch, arch_vector)
model = set_named_arch(model, arch)
model(input_x)

# print("FLOPS:")
# input_size = [10, 3, 28, 28]
# print(get_flops_and_memory(model, input_size))
# set_signle(model)
# print(model)
# model(input_x)
# print("FLOPS:")
# print(get_flops_and_memory(model, input_size))
