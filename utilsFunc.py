import time
import functools
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

#Decorator Function
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

def plot_kernels1D(tensor,FFT=False,fe=8000,plotName=False):
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
  plt.show()

def plot_kernels2D(tensor,plotName=False,transpose=False):

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
  plt.show()


def train(storage,exp_i=0):
    storage['model'][exp_i].train()
    for batch_idx, (data, target) in enumerate(storage['train_loader'][exp_i]):

        data = data.to(storage['device'])
        target = target.to(storage['device'])

        # apply transform and model on whole batch directly on device
        data = storage['transform'][exp_i](data)
        output = storage['model'][exp_i](data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = storage['lossFunc'](output.squeeze(), target)

        storage['optimizer'][exp_i].zero_grad()
        loss.backward()
        storage['optimizer'][exp_i].step()

        # print training stats
        if batch_idx % storage['log_interval'][exp_i] == 0:
            print(
                f"Train Epoch: {storage['epoch']} [{batch_idx * len(data)}/{len(storage['train_loader'][exp_i].dataset)}"
                f" ({100. * batch_idx / len(storage['train_loader'][exp_i]):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        storage['pbar'].update(storage['pbar_update'][exp_i])
        # record loss
        storage['losses_train'][storage['model'][exp_i]].append(loss.item())

def validation(storage,exp_i=0):
    storage['model'][exp_i].eval()
    correct = 0
    for data, target in storage['validation_loader'][exp_i]:
        data = data.to(storage['device'])
        target = target.to(storage['device'])

        # apply transform and model on whole batch directly on device
        data = storage['transform'][exp_i](data)
        output = storage['model'][exp_i](data)

        correct += storage['metrics'](output,target)

        # update progress bar
        storage['pbar'].update(storage['pbar_update'])

    print(
        f"\nTest Epoch: {storage['epoch']}\tAccuracy: {correct}/{len(storage['validation_loader'][exp_i].dataset)} "
        f"({100. * correct / len(storage['validation_loader'][exp_i].dataset):.0f}%)\n")




def showResult(model):
    FirstLayerWeights = model.conv1.weight.detach().cpu().numpy()
    plot_kernels1D(FirstLayerWeights)
    plot_kernels1D(FirstLayerWeights, True)
    SecondLayerWeights =model.conv2.weight.detach().cpu().numpy()
    plot_kernels1D(SecondLayerWeights)
    plot_kernels1D(SecondLayerWeights, True)

currentOrLast = lambda c,lst: c if c <len(lst) else -1
if __name__=='__main__':
    with timeThat() :
        print('hello')
