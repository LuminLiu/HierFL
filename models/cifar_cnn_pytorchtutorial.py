# include a complete network, shared network and task specific part

import torch.nn as nn
import torch.nn.functional as F


class cifar_cnn_pytorchtutorial(nn.Module):
    """
    The full version of the model
    """
    def __init__(self, input_channels = 3,output_channels = 10 ):
        super(cifar_cnn_pytorchtutorial, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels= 6,
                               kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.view(-1, 16 *5 *5)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h


class cifar_cnn_pytorchtutorial_shared(nn.Module):
    """
        shared layers (feature layers)
        """

    def __init__(self, input_channels):
        super(cifar_cnn_pytorchtutorial_shared, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.view(-1, 16 * 5 * 5)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

    def feature_out_dim(self):
        return 84

class cifar_cnn_pytorchtutorial_specific(nn.Module):
    """
    task specific layers
    """
    def __init__(self, input_channels, output_channels):
        super(cifar_cnn_pytorchtutorial_specific, self).__init__()
        self.fc = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        h = self.fc(x)
        return h