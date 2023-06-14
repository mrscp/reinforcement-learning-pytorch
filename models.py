import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelTorch(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(ModelTorch, self).__init__()
        self.input_shape = input_shape
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_shape[2], out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2)
        )
        self.layer2 = nn.Linear(4096, 8)
        self.layer3 = nn.Linear(8, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_shape[2], self.input_shape[0], self.input_shape[1]))
        x = F.relu(self.layer1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
