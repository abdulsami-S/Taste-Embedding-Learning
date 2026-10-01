import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, d_in=384, d_hidden=512, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)