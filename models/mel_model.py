import torch
import torch.nn as nn
import torch.nn.functional as F


class mel_model(nn.Module):
    def __init__(self, input_shape=0, n_channel=64,n_output=35,debug=False):
        super().__init__()

        self.dropout=nn.Dropout(0.5)
        self.conv1 =  nn.Conv2d(1, n_channel, kernel_size=(20,8), stride=(1,3))

        self.seq = nn.Sequential(nn.ReLU(),
                                    self.dropout,
                                    nn.MaxPool2d((1,3))
        )

        self.conv2 = nn.Sequential( nn.Conv2d(n_channel, n_channel, kernel_size=(10,4), stride=(1,1)),
                                    nn.ReLU(),
                                    self.dropout
        )

        self.flatten = nn.Flatten()

        # Get the shape of the resulting tensor
        size_tensor = torch.zeros(input_shape)
        if size_tensor.ndim==3 :
            size_tensor=size_tensor[None,:]
        size_tensor = self.conv1(size_tensor)
        size_tensor = self.seq(size_tensor)
        size_tensor = self.conv2(size_tensor)
        size_tensor = self.flatten(size_tensor).shape[-1]
        if debug:
            print('size of flatten tensor', size_tensor)


        self.fc=nn.Sequential(
            nn.Linear(size_tensor, int(n_channel / 2)),
            nn.ReLU(),
            nn.Linear(int(n_channel / 2), n_channel * 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(2 * n_channel, n_output)

        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.seq(x)
        x = self.conv2(x)
        x = self.flatten(x)

        x=self.fc(x)
        return F.log_softmax(x,dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=='__main__':
    x=40
    y=101
    test=torch.zeros((256,1,x,y))
    print(test.shape)
    model=mel_model(test.shape,debug=True)
    result = model(test)
    print('test passé')