import torch.nn as nn
import torch.nn.functional as F


class spectrogram_model(nn.Module):
    def __init__(self, input_shape, n_channel=32,stride=2,n_output=35):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=10, stride=stride)
        self.pool1 = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(n_channel, 2*n_channel, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(1344, 2 * n_channel)
        self.fc2 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x=self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x=self.dropout(x)

        x = self.conv3(x)
        x=self.flatten(x)

        x=self.fc1 (x)
        x=self.fc2 (x)
        return F.log_softmax(x,dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    pass