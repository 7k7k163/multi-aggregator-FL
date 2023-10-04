from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
