import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Dedykowana głowica dla pojedynczego zadania ADMET."""

    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h),
                nn.LeakyReLU(0.01),  # Aktywacja LeakyReLU zgodnie z eq. 2 [cite: 252, 254]
                nn.Dropout(0.3)  # Dropout p=0.3 [cite: 256, 261]
            ])
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MTAN_ADMET(nn.Module):
    """Główny model Multi-Task Adaptive Network."""

    def __init__(self, input_dim=512, shared_dims=[1024, 512]):
        super().__init__()
        # Warstwy współdzielone (Shared Layers) [cite: 212]
        shared = []
        curr_dim = input_dim
        for h in shared_dims:
            shared.extend([
                nn.Linear(curr_dim, h),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.3)
            ])
            curr_dim = h
        self.shared_net = nn.Sequential(*shared)

        # 24 dedykowane bloki dla każdego punktu końcowego [cite: 242, 262]
        self.heads = nn.ModuleList([TaskHead(curr_dim) for _ in range(24)])

    def forward(self, x):
        features = self.shared_net(x)
        return [head(features) for head in self.heads]