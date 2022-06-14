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
        output = storage['model'][exp_i](data)

        correct += storage['metrics'](output, target)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = storage['lossFunc'](output.squeeze(), target)
        if not validation :
            storage['optimizer'][exp_i].zero_grad()
            loss.backward()
            storage['optimizer'][exp_i].step()

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
        print(
            f"\nvalidation Epoch: {storage['epoch']}\tAccuracy: {correct}/{len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])].dataset)} "
            f"({100. * correct / len(storage[mode+'_loader'][currentOrLast(exp_i,storage[mode+'_loader'])].dataset):.0f}%)\n")




def test(storage,exp_i=0):
    storage['model'][exp_i].eval()
    correct = 0
    for data, target in storage['test_loader'][currentOrLast(exp_i,storage['test_loader'])]:
        data = data.to(storage['device'])
        target = target.to(storage['device'])

        # apply transform and model on whole batch directly on device
        data = storage['transform'][exp_i](data)
        output = storage['model'][exp_i](data)

        correct += storage['metrics'](output,target)

        # update progress bar

        storage['pbar'].update(storage['pbar_update'][currentOrLast(exp_i, storage['pbar_update'])])
    print(
        f"\nTest Epoch: {storage['epoch']}\tAccuracy: {correct}/{len( storage['test_loader'][currentOrLast(exp_i, storage['test_loader'])].dataset)} "
        f"({100. * correct / len(    storage['test_loader'][currentOrLast(exp_i, storage['test_loader'])].dataset):.0f}%)\n")


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

currentOrLast = lambda c,lst: c if c <len(lst) else -1
if __name__=='__main__':
    with timeThat() :
        print('hello')
