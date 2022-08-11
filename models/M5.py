import torch
import torch.nn as nn
import torch.nn.functional as F

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.temp =nn.Sequential(
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4))

        self.second_layer= nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        self.fc1 = nn.Linear(2 * n_channel, n_output)


    def forward(self, x):
        x = self.conv1(x)
        x= self.temp(x)
        x = self.second_layer(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    a=M5()
    input=torch.zeros((1,1,8000))
    output=a(input)
    print('')
