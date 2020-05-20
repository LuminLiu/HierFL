# include a complete network, shared network and task specific part

import torch.nn as nn
import torch.nn.functional as F

class femnist_cnn_leaf(nn.Module):
    """
    The full version of the model
    """
    def __init__(self, input_channels = 3,output_channels = 10 ):
        super(femnist_cnn_leaf, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels= 32,
                               kernel_size=5,
                               padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, output_channels)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.view(-1, 7 *7 *64)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h


class femnist_cnn_leaf_shared(nn.Module):
    """
        shared layers (feature layers)
        """

    def __init__(self, input_channels):
        super(femnist_cnn_leaf_shared, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=32,
                               kernel_size=5,
                               padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, output_channels)

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = h.view(-1, 7 * 7 * 64)
        h = F.relu(self.fc1(h))
        # h = F.relu(self.fc2(h))
        return h

    def feature_out_dim(self):
        return 2048

class femnist_cnn_leaf_specific(nn.Module):
    """
    task specific layers
    """
    def __init__(self, input_channels, output_channels):
        super(femnist_cnn_leaf_specific, self).__init__()
        self.fc = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        h = self.fc(x)
        return h