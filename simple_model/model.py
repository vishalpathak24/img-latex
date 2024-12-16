import io
import torch

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 10
