import random

import torch
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
            nn.Conv1d(n_channel, n_channel, kernel_size=int(kernel_size/2),stride=stride,dilation=dilation),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4)),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=int(kernel_size/4),stride=stride,dilation=dilation),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(int(stride/4)),
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

MAX_DEPTH=10
def try_param(PDM_factor=10,n_channel=10,stride=4,kernel_size=1600,dilation=1,depth=0):
    if depth > MAX_DEPTH:
        print('stopping the search :\n','try_param max depth reach')
        return False
    depth+=1

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 16000 * PDM_factor
    in_test = torch.zeros((1, 1, input_size)).to(device )

    try :
        a = PDM_model(stride=stride, n_channel=n_channel, kernel_size=kernel_size, dilation=dilation).to( device)
        output = a(in_test)
        print('test passed,', a.count_parameters(), 'parameters')
        return (stride, n_channel, kernel_size, dilation)
    except Exception as e:
        if kernel_size <MIN_KERNEL_SIZE :
            print('stopping the search :\n','min kernel size reached')
            return False
        if OUTPUT_SIZE in str(e)  :
            #We should increase kernel_size
            print(e)
            print('trying to increase kernel size')
            return try_param(stride=stride, n_channel=n_channel,
                kernel_size=kernel_size*2, dilation=dilation,depth=depth)
        elif KERNEL_SIZE in str(e) :
            #We should reduce the kernel size
            print(e)
            print('trying to reduce kernel size')
            return try_param(stride=stride, n_channel=n_channel,
                kernel_size=int(kernel_size / 2), dilation=dilation,depth=depth)

        else :
            print('stopping the search :\n',e)
            return False

OUTPUT_SIZE='Output size is too small'
KERNEL_SIZE="Kernel size can't be greater than actual input size"
MIN_KERNEL_SIZE=10

if __name__=='__main__':
    for stride in [i for i in range(1,50,10)]:
        try_param(dilation=stride)
