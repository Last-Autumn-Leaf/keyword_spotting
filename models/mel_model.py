import torch
import torch.nn as nn
import torch.nn.functional as F


class mel_model(nn.Module):
    def __init__(self, input_shape=0, n_channel=64,n_output=35):
        super().__init__()

        self.dropout=nn.Dropout(0.5)
        self.conv1 = nn.Sequential( nn.Conv2d(1, n_channel, kernel_size=(20,8), stride=(1,3)),
                                    nn.ReLU(),
                                    self.dropout,
                                    nn.MaxPool2d((1,3))
        )

        self.conv2 = nn.Sequential( nn.Conv2d(n_channel, n_channel, kernel_size=(10,4), stride=(1,1)),
                                    nn.ReLU(),
                                    self.dropout
        )

        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(5376, int(n_channel/2))
        self.fc2 = nn.Linear(int(n_channel/2), n_channel*2)
        self.fc3 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x=self.flatten(x)

        x=self.fc1 (x)
        x = F.relu(x)
        x=self.fc2 (x)
        x = F.relu(x)
        x=self.dropout(x)
        x=self.fc3 (x)
        return F.log_softmax(x,dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    x=40
    y=101
    test=torch.zeros((256,1,x,y))
    print(test.shape)
    model=mel_model(test.shape)
    result = model(test)
    print('test passé')