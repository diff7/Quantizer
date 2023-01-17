import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 8, 5, stride=5, bias=True)
        # alpha_shape = [1, 8] + [1] * (self.conv1.weight.dim() - 2)
        # self.alpha_one = nn.Parameter(torch.ones(*alpha_shape))
        self.conv2 = nn.Conv2d(8, 8, 3, stride=1, bias=True)
        # alpha_shape = [1, 8] + [1] * (self.conv1.weight.dim() - 2)
        # self.alpha_two = nn.Parameter(torch.ones(*alpha_shape))

        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)  # * self.alpha_one
        x = F.relu(x)
        x = self.conv2(x)  # * self.alpha_two
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return {"preds": output}
