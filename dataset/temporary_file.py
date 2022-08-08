import os

import torch
import numpy as np
from helper.utilsFunc import tensorToBytes, BytesToTensor

a=torch.tensor([1,1,1,0,0,0,1,0])
b=torch.tensor([0,0,0])
c=torch.tensor([1,1,1])


file = open("sample.bin", "wb")
x_packed = tensorToBytes(a)
file.write(x_packed)
x_unpacked = BytesToTensor(x_packed)
file.close()

#--------
offset=0
size=8
with open("sample.bin", "rb") as f:
        f.seek(offset*size)
        byte = f.read(size)


print(BytesToTensor(byte))