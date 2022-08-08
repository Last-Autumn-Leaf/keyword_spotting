import os

import torch

from helper.utilsFunc import tensorToBytes, BytesToTensor

a=torch.tensor([1,1,1])
b=torch.tensor([0,0,0])
c=torch.tensor([1,1,1])


file = open("sample.bin", "ab")
file.write(tensorToBytes(a))
file.write(tensorToBytes(b))
file.write(tensorToBytes(c))
file.close()

#--------
offset=1
size=3
with open("sample.bin", "rb") as f:
        f.seek(offset*size)
        byte = f.read(size)


print(BytesToTensor(byte))