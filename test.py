import torch

from QSB.qconfig import QConfig
from QSB.tools import (
    replace_modules,
    set_signle,
    get_flops_and_memory,
    prepare_and_get_params,
)

# from models.simple import SimpleCNN
from models.IMDN.architechture import IMDN

qconfig = QConfig()
# model = SimpleCNN()
model = IMDN()

input_x = torch.randn(10, 3, 28, 28)
model(input_x)

model, main_params, alpha, alpha_names = prepare_and_get_params(
    model, qconfig, verbose=True
)
print("#" * 10)
print("\n \n", alpha_names)
print(len(alpha_names))

# model(input_x)

# print("FLOPS:")
# input_size = [10, 3, 28, 28]
# print(get_flops_and_memory(model, input_size))
# set_signle(model)
# print(model)
# model(input_x)
# print("FLOPS:")
# print(get_flops_and_memory(model, input_size))
