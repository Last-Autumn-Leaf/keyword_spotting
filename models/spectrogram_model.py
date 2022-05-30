import torch
import torch.nn as nn
import torch.nn.functional as F


class spectrogram_model(nn.Module):
    def __init__(self, input_shape, n_channel=64,stride=1,n_output=35):
        super().__init__()

        x=int(input_shape[-2] /5)
        y=int(input_shape[-1] /5)
        self.conv1 = nn.Conv2d(1, n_channel, kernel_size=(x,y), stride=stride)
        self.pool1 = nn.MaxPool2d(2)
        x=int(x/2)
        y=int(y/2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(x,y))
        self.dropout=nn.Dropout(0.5)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(151552, int(n_channel/2))
        self.fc2 = nn.Linear(int(n_channel/2), n_channel*2)
        self.fc3 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x=self.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x=self.dropout(x)

        x=self.flatten(x)

        x=self.fc1 (x)
        x=self.fc2 (x)
        x=self.dropout(x)
        x=self.fc3 (x)
        return F.log_softmax(x,dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    x=241
    y=101
    test=torch.zeros((256,1,x,y))
    print(test.shape)
    model=spectrogram_model(test.shape)
    result = model(test)
    print('test passé')