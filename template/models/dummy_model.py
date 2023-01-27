import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, padding="same"), nn.ReLU(), nn.Conv1d(12, 1, kernel_size=3, padding="same")
        )

    def forward(self, x):
        if x.dim() == 2:
            y = self.cnn(x.unsqueeze(1))
        elif x.dim() == 3:
            y = self.cnn(x)
        else:
            pass
        return y
