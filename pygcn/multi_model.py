import torch.nn as nn
import torch

class MultiModel(nn.Module):

    def __init__(self, modeltype, n_models, *args, **kwargs):
        super().__init__()
        models = [modeltype(*args, **kwargs) for _ in range(n_models)]
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        outputs = [m(*args, **kwargs) for m in self.models]
        return torch.stack(outputs)