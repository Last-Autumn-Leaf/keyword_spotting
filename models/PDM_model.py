
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import  firwin
class PDM_model(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32,kernel_size=100,dilation=1,maxpool=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=kernel_size, stride=stride,dilation=dilation)
        self.init_low_pass_conv1d()

        self.temp = nn.Sequential(
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),)
        self.max_p=nn.MaxPool1d(maxpool)


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

        batch_size, in_channels, len_input_sequence = x.shape
        low_pass_output_shape = (batch_size, in_channels, len_input_sequence-self.low_pass_kernel_size+1)
        x = F.conv1d(x.view(-1, 1, len_input_sequence), self.low_pass_kernel).view(*low_pass_output_shape)

        x = self.temp(x)
        x = self.max_p(x)
        x = self.second_layer(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

    def init_low_pass_conv1d(self):
        WAV_SAMPLE_RATE = 16000
        PDM_FACTOR = 20
        PDM_SAMPLE_RATE = WAV_SAMPLE_RATE * PDM_FACTOR
        PDM_NYQUIST = PDM_SAMPLE_RATE / 2

        PDM_LOW_PASS_CUTOFF = (WAV_SAMPLE_RATE / 2)
        PDM_LOW_PASS_CUTOFF_NORM = PDM_LOW_PASS_CUTOFF / PDM_NYQUIST

        # PDM_LOW_PASS_N_TAPS corresponds to torch.conv1d: kernel_size
        # Note : this controls the sharpness of the low-pass filter cutoff
        # higher value will result in sharper cutoff, but more computation.
        # Since the majority of speech energy is typically < 4 kHz,
        # high sharpness is likely not terribly important
        PDM_LOW_PASS_N_TAPS = 128
        lowpass_filter_weights = firwin(PDM_LOW_PASS_N_TAPS, PDM_LOW_PASS_CUTOFF_NORM, pass_zero='lowpass')
        kernel = torch.tensor(lowpass_filter_weights, dtype=torch.float32)[None, None]
        
        self.low_pass_kernel_size = PDM_LOW_PASS_N_TAPS

        self.low_pass_kernel = kernel.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

MAX_DEPTH=10
def try_param(PDM_factor=10,n_channel=10,stride=4,kernel_size=1600,dilation=1,maxpool=4,depth=0):
    if depth > MAX_DEPTH:
        print('stopping the search :\n','try_param max depth reach')
        return False
    depth+=1

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 16000 * PDM_factor
    in_test = torch.zeros((1, 1, input_size)).to(device )

    try :
        a = PDM_model(stride=stride, n_channel=n_channel, kernel_size=kernel_size, dilation=dilation,maxpool=maxpool).to( device)
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
    PDM_factor=20
    stride = 1
    n_channel = 4
    kernel_size = 80
    dilation = 40

    a = PDM_model(kernel_size=kernel_size, stride=stride,dilation=dilation,n_channel=n_channel)
    input = torch.zeros((1, 1, 16000*PDM_factor))
    output = a(input)
    print(a.count_parameters())
