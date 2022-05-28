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

if __name__=='__main__':
    with timeThat() :
        print('hello')
