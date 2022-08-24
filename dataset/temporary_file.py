
import torch

from scipy.signal import  firwin
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
kernel = torch.tensor(lowpass_filter_weights, dtype=torch.float32)

n_channel=32

true_kernel=torch.zeros((n_channel,n_channel,PDM_LOW_PASS_N_TAPS))

for i in range(n_channel):
    for j in range(n_channel):
        true_kernel[i,j]=kernel

conv2=torch.nn.Conv1d(n_channel,n_channel,PDM_LOW_PASS_N_TAPS)
conv2.weight=torch.nn.parameter.Parameter(true_kernel)
conv2.bias=torch.nn.parameter.Parameter(torch.zeros((n_channel)))
conv2.requires_grad_(False)
print(conv2.weight)

#batch size,n_channel,kernel_size
test=torch.zeros((1,32,200))
test2=conv2(test)