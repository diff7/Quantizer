import torch
from qconfig import QConfig
from replace import replace_modules, set_signle, get_alphas
from models.simple import SimpleCNN


qconfig = QConfig()
model = SimpleCNN()

input_x = torch.randn(10, 1, 28, 28)
model(input_x)

replace_modules(model, qconfig, verbose=True)
print(model)
model(input_x)
print(get_alphas(model))
set_signle(model)
print(model)
model(input_x)
