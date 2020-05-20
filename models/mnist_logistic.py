import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim = 1, output_dim = 10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim*28*28, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)
        outputs = self.linear(x)
        return outputs
