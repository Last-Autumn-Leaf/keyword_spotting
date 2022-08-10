import sys

import time
import zipfile

import numpy as np
import torch
import os
from torch.utils.data import Dataset

from dataset.subsetSC import SubsetSC
from helper.utilsFunc import PdmTransform
from helper.utilsFunc import tensorToBytes, BytesToTensor

import pathlib

white_list_mode=['training','testing','validation']
class SubsetPDM(Dataset):
    def __init__(self,pdm_factor:int=20,fe:int=16000,subset:str=white_list_mode[0],
                 root=pathlib.Path('./')):
        self.pdm_factor=pdm_factor
        self.fe=fe
        self.size=int(pdm_factor * fe /8)
        if subset in white_list_mode :
            self.mode=subset
        else :
            raise Exception('unrecognized mode' + str(subset))
        self.path=pathlib.Path(root)
        self.tensor_path=self.path / "PDM_{}_{}_tensor.bin".format(str(self.pdm_factor),self.mode)
        self.label_path= self.path / "PDM_{}_{}_label.txt".format(str(self.pdm_factor),self.mode)

        if not self.tensor_path.is_file() or not self.label_path.is_file() :
            print('files not found, trying to unzip them from',pathlib.Path.cwd())
            zip_path=pathlib.Path.cwd() / 'PDM_{}_{}.zip'.format(pdm_factor,subset)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            if self.tensor_path.is_file() and self.label_path.is_file():
                raise Exception('dataset files not found')

        self.getLabels()
        f = open(self.label_path, "r")
        j=0
        for j, line in enumerate(f):
            ...
        f.close()
        self.length = j+1

    def label_to_index(self,word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def index_to_label(self,index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def getLabels(self):
        f = open(self.label_path, "r")
        self.labels = sorted(list(set(line.replace('\n','') for j, line in enumerate(f))))
        f.close()
        return self.labels

    def collate_fn(self,batch):
        tensors, targets = [], []
        # Gather in lists, and encode labels as indices
        for waveform,label in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        tensors = torch.stack(tensors)
        targets = torch.stack(targets)
        return tensors, targets

    def to (self,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device=device
        return self
    def __len__(self): return self.length
    def __getitem__(self, i):

        #TODO : it is possible to keep the file open in the initializer
        byte=b''
        with open(self.tensor_path, "rb") as f:
            f.seek(i * self.size)
            byte = f.read(self.size)
        label=''
        with open(self.label_path, "r")  as f:
            for j, line in enumerate(f):
                if j == i:
                    label=line.replace('\n','')
                    break

        return BytesToTensor(byte).to(self.device),label


def setupPDMtoText(pdm_factor=20,mode='training',root= pathlib.Path('./'),
                   device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   bsize=5000):
    if mode not in white_list_mode:
        raise Exception('unrecognized mode: '+str(mode))
    fe = 16000


    PDM_TRAMSFORM = PdmTransform(pdm_factor=pdm_factor, signal_len=fe,
                                 orig_freq=fe).to(device)

    subset = SubsetSC(mode, root)
    print(mode,'subset size :',len(subset))
    n=len(subset)

    # unlink allow us to delete the file if it already exists
    tensor_path= root/"PDM_{}_{}_tensor.bin".format(str(pdm_factor),mode)
    tensor_path.unlink(missing_ok=True)
    label_path=root/"PDM_{}_{}_label.txt".format(str(pdm_factor), mode)
    label_path.unlink(missing_ok=True)

    file_tensor = open(tensor_path, "ab")
    file_labels = open(label_path, "a")
    print('Saving dataset at :\n',file_tensor,'\n',file_labels)

    def pad_sequence(x, n_samples=16000):
        n_pad = n_samples - x.shape[-1]
        if n_pad != 0:
            x = torch.nn.functional.pad(x, (0, n_pad), value=0.0)
        return x

    i=0
    while i<n:
        end = i+bsize if i+bsize<n else n
        samples=[]
        labels =''

        for ii in range(i, end) :
            samples.append(pad_sequence(subset[ii][0]))
            labels+=subset[ii][2]+'\n'
        i=end

        samples = torch.stack(samples)
        samples = pad_sequence(samples.to(device))
        # Resample et PDM transform
        samples = PDM_TRAMSFORM(samples)
        waveform=tensorToBytes(samples)

        file_tensor.write(waveform)
        file_labels.write(labels)

        print('itération ',i,'/'+str(n),end='\r')


    file_tensor.close()
    file_labels.close()

    return tensor_path,label_path


if __name__=='__main__':
    if len(sys.argv) >1 :
        print('job index',sys.argv[1])
        index=int(sys.argv[1])
        pdm_factor=20
        mode=white_list_mode[index]
        root=pathlib.Path('../')
        if 'SLURM_TMPDIR' in os.environ:
            root = pathlib.Path(os.environ['SLURM_TMPDIR'])
            print('Compute Canada detected, root set to', root)

        #setupPDMtoText(pdm_factor=pdm_factor,mode=mode,root=root,bsize=5000)

        print('testing')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = SubsetPDM(subset=mode, root=root).to(device)
        print(a[1])


