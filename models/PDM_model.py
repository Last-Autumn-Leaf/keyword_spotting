import torch.nn as nn
import torch.nn.functional as F

class PDM_model(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32,kernel_size=100,dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=kernel_size, stride=stride,dilation=dilation)

        self.second_layer= nn.Sequential(
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4)),
            nn.Conv1d(n_channel, n_channel, kernel_size=int(kernel_size/10),stride=stride,dilation=dilation),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4)),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=int(kernel_size/10),stride=stride,dilation=dilation),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4)),
            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=int(kernel_size/10),stride=stride,dilation=dilation),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4))
        )

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.second_layer(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    a=PDM_model()
    l = [module for module in a.modules() if not isinstance(module, nn.Sequential)]
    b = {}
    layers_to_record = [nn.Conv1d, nn.Conv2d]
    index = 0
    for lay in l:
        if type(lay) in layers_to_record:
            b['conv ' + str(index)] = str(lay)
            index += 1
    print(b)
