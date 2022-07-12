import time
import functools
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

#Decorator Function
import torch
import torchaudio


def timeThis(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start=time.time()
        a= func(*args, **kwargs)
        end = time.time()
        print('Finished in ',timedelta(seconds=end-start))
        return a
    return wrapper

@timeThis
def test(a='test'):
    print(a)

'''#Implementing a Context Manager as a Class
class timeThat(object):
    def __enter__(self):
        self.start=time.time()
    def __exit__(self, type, value, traceback):
        if type==None :
            self.end = time.time()
            print('Finished in ', round(self.end - self.start, 1), 's')
            return True
        else : return type, value, traceback'''


from contextlib import contextmanager
@contextmanager
def timeThat(name=''):
    try:
        start = time.time()
        yield ...
    finally:
        end = time.time()
        print(name+' finished in ',timedelta(seconds=end-start))

def plotFFT(tensor,fe=8000,ax=None):
  tensor-= tensor.mean()
  if ax ==None:
    plt.figure()
  spectrum = fft.rfft(tensor)
  freq = fft.rfftfreq(len(tensor),d=1/fe)
  if ax ==None:
    plt.plot(freq,abs(spectrum))
    plt.show()
  else:
    ax.plot(freq,abs(spectrum))

def plot_kernels1D(tensor,FFT=False,fe=8000,plotName=False,save=None):
  if tensor.ndim==2:
    tensor=tensor[:,None,:]

  if not tensor.ndim==3:
      raise Exception("assumes a 3D tensor")
  num_kernels = tensor.shape[0]
  sep=int(np.ceil(np.sqrt(num_kernels)))
  fig = plt.figure(figsize=(sep, sep))
  for i in range(num_kernels):
    a=fig.add_subplot(sep, sep, i+1)
    if FFT :
      plotFFT(tensor[i,0],fe,ax=a)
    else:
      a.plot(tensor[i,0])
    a.axis('off')
    if plotName:
      a.set_title('kernel'+str(i))
  if save != None :
      plt.savefig(save+ '.png')
  plt.show()

def plot_kernels2D(tensor,plotName=False,transpose=False,save=None):
  if tensor.ndim==3 :
      tensor=tensor[:,None,:]

  if not tensor.ndim==4:
      raise Exception("assumes a 4D tensor")
  num_kernels = tensor.shape[0]
  sep=int(np.ceil(np.sqrt(num_kernels)))
  fig = plt.figure(figsize=(sep, sep),dpi=500)
  for i in range(num_kernels):
    a=fig.add_subplot(sep, sep, i+1)
    a.imshow(tensor[i,0].T if transpose else tensor[i,0])
    a.axis('off')
    if plotName:
      a.set_title('kernel'+str(i))
  if save != None :
      plt.savefig(save+ '.png')
  plt.show()


def train(storage,exp_i=0,validation=False):
    if not validation: storage['model'][exp_i].train()
    else : storage['model'][exp_i].eval()
    correct = 0
    mode='val' if validation else 'train'
    for batch_idx, (data, target) in enumerate(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])]):
        data = data.to(storage['device'])
        target = target.to(storage['device'])
        # apply transform and model on whole batch directly on device
        data = storage['transform'][exp_i](data)
        #data = data.to(storage['device'])
        output = storage['model'][exp_i](data)
        correct += storage['metrics'](output, target)


        loss = storage['lossFunc'](output.squeeze(), target)

        if not validation :
            storage['optimizer'][exp_i].zero_grad()
            loss.backward()
            storage['optimizer'][exp_i].step()
            storage['writer'].add_scalar(storage['exp_name']+'/Loss/Train ' + storage['current_model'], loss.item(),storage['train_index'])
            storage['train_index']+=1
        else :
            storage['writer'].add_scalar(storage['exp_name']+'/Loss/Validation ' + storage['current_model'], loss.item(),
                                         storage['val_index'])
            storage['val_index'] += 1

        # print training stats
        if batch_idx % storage['log_interval'] == 0:
            print(
                mode+f" Epoch: {storage['epoch']} [{batch_idx * len(data)}/{len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])].dataset)}"
                f" ({100. * batch_idx / len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])]):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        storage['pbar'].update(storage['pbar_update'][currentOrLast(exp_i,storage['pbar_update'])])
        # record loss
        storage['losses_'+mode][exp_i].append(loss.item())



    storage['accuracy']=100. * correct / len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])].dataset)
    if validation:
        storage['writer'].add_scalar(storage['exp_name']+'/Accuracy/Validation ' + storage['current_model'], storage['accuracy'], storage['acc_index'])
        storage['acc_index'] += 1
        print(
            f"\nvalidation Epoch: {storage['epoch']}\tAccuracy: {correct}/{len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])].dataset)} "
            f"({storage['accuracy']:.0f}%)\n")


def test(storage,exp_i=0):
    storage['model'][exp_i].eval()
    correct = 0
    for data, target in storage['test_loader'][currentOrLast(exp_i,storage['test_loader'])]:
        data = data.to(storage['device'])
        target = target.to(storage['device'])

        # apply transform and model on whole batch directly on device
        data = storage['transform'][exp_i](data)
        #data = data.to(storage['device'])
        output = storage['model'][exp_i](data)

        correct += storage['metrics'](output,target)

        # update progress bar
        storage['pbar'].update(storage['pbar_update'][currentOrLast(exp_i, storage['pbar_update'])])
    correct_percent=100. * correct / len(    storage['test_loader'][currentOrLast(exp_i, storage['test_loader'])].dataset)
    print(
        f"\nTest Epoch: {storage['epoch']}\tAccuracy: {correct}/{len( storage['test_loader'][currentOrLast(exp_i, storage['test_loader'])].dataset)} "
        f"({correct_percent:.0f}%)\n")
    storage['writer'].add_scalar('prediction', correct_percent)


def space_frequency(image):
    freqx = np.fft.fftfreq(image.shape[0])
    freqy = np.fft.fftfreq(image.shape[1])
    image_fft = np.fft.fft2(image)


def showResult(model):
    FirstLayerWeights = model.conv1.weight.detach().cpu().numpy()
    plot_kernels1D(FirstLayerWeights)
    plot_kernels1D(FirstLayerWeights, True)
    SecondLayerWeights =model.conv2.weight.detach().cpu().numpy()
    plot_kernels1D(SecondLayerWeights)
    plot_kernels1D(SecondLayerWeights, True)



class PdmTransform(torch.nn.Module):
    def __init__(self, orig_freq: int = 16000,pdm_factor: int = 48,signal_len:int=16000):
        super(PdmTransform, self).__init__()
        self.PDM_transform=torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=int(np.round(signal_len * pdm_factor)))

    def to(self,device):
        self.device=device
        self.PDM_transform=self.PDM_transform.to(device)
        return self

    def pdm(self,x):
        if x.ndim ==1 :
            n = len(x)
        else:
            n = len(x[-1])
        y = torch.zeros_like(x).to(self.device)
        shape=[* x.shape]
        shape[-1]+=+1
        error = torch.zeros(shape).to(self.device)
        for i in range(n):
            idx = (np.s_[:],) * (x.ndim-1) + (i,)
            y[idx] = torch.where( x[idx] >= error[idx] ,
                                  torch.ones(shape[:-1]).to(self.device),
                                  torch.zeros(shape[:-1]).to(self.device))
            error[(np.s_[:],) * (x.ndim-1) + (i+1,)] = y[idx] - x[idx] + error[idx]
        return y, error[:n]

    def __call__(self, samples):
        upsampled_samples = self.PDM_transform(samples)
        samples +=1
        samples /=2
        # upsampled_samples = resample(samples, n_pdm_samples)
        pdm_samples, pdm_error = self.pdm(upsampled_samples)
        return pdm_samples

    def __repr__(self):
        return "custom PDM transform, does the rescale and the transform"



currentOrLast = lambda c,lst: c if c <len(lst) else -1
if __name__=='__main__':
    with timeThat() :
        print('hello')