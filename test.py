import torch

from QSB.qconfig import QConfig
from QSB.tools import (
    replace_modules,
    set_signle,
    get_alphas_and_set_grad_true,
    get_flops_and_memory,
    prepare_and_get_params,
)

from models.simple import SimpleCNN

qconfig = QConfig()
model = SimpleCNN()

input_x = torch.randn(10, 1, 28, 28)
model(input_x)

model, main_params, alpha = prepare_and_get_params(model, qconfig, verbose=True)
model(input_x)

print("FLOPS:")
input_size = [10, 1, 28, 28]
print(get_flops_and_memory(model, input_size))
set_signle(model)
print(model)
model(input_x)
print("FLOPS:")
print(get_flops_and_memory(model, input_size))
