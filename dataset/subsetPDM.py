import sys

import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

from dataset.subsetSC import SubsetSC
from helper.utilsFunc import PdmTransform
from helper.utilsFunc import tensorToBytes, BytesToTensor
import linecache

white_list_mode=['training','testing','validation']
class SubsetPDM(Dataset):
    def __init__(self,pdm_factor:int=20,fe:int=16000,mode:str=white_list_mode[0]):
        self.pdm_factor=pdm_factor
        self.size=pdm_factor * 16000
        if mode in white_list_mode :
            self.mode=mode
        else :
            raise Exception('unrecognized mode'+str(mode))

    def setData(self,data):
        self.data=data

    def setTarget(self,target):
        self.target=target

    def label_to_index(self,word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))

    def index_to_label(self,index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def getLabels(self):
        self.labels = sorted(list(set(datapoint for datapoint in self.target)))
        return self.labels

    def collate_fn(self,batch):

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform,label in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        targets = torch.stack(targets)

        return tensors, targets

    def to (self,device):
        self.device=device
        return self
    def __len__(self): return len(self.target)
    def __getitem__(self, i):

        #TODO : it is possible to keep the file open in the initializer
        with open("PDM_{}_{}_tensor.bin".format(str(self.pdm_factor),self.mode), "rb") as f:
            f.seek(i * self.size)
            byte = f.read(self.size)

        with open("PDM_{}_{}_label.txt".format(str(pdm_factor), self.mode), "r")  as f:
            for j, line in enumerate(f):
                if j == i:
                    label=line.replace('\n','')
                    break


        #keep the whole file in memory so not good, better use an itterator
        #label =linecache.getline("PDM_{}_{}_label.txt".format(str(pdm_factor), mode), i)
        #TODO : cast label to int via label_to_index
        return BytesToTensor(byte).to(self.device),label


def setupPDMtoText(pdm_factor=20,mode='training'):
    if mode not in white_list_mode:
        raise Exception('unrecognized mode: '+str(mode))
    fe = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device ="cpu"
    PDM_TRAMSFORM = PdmTransform(pdm_factor=pdm_factor, signal_len=fe,
                                 orig_freq=fe).to(device)

    root ='./'
    if 'SLURM_TMPDIR' in os.environ:
        root = os.environ['SLURM_TMPDIR']
        print('Compute Canada detected, root set to', root)

    subset = SubsetSC(mode, root)
    print(mode,'subset size :',len(subset))
    n=len(subset)

    label,waveform='',b''
    file_tensor = open("PDM_{}_{}_tensor.bin".format(str(pdm_factor),mode), "ab")
    file_labels = open("PDM_{}_{}_label.txt".format(str(pdm_factor), mode), "a")
    for i in range(n):
        temp = subset[i]
        label = temp[2] +'\n'
        temp = subset.pad_sequence(temp[0].to(device))
        # Resample et PDM transform
        temp = PDM_TRAMSFORM(temp)

        waveform+=tensorToBytes(temp)

        file_tensor.write(waveform)
        file_labels.write(label)


    file_tensor.close()
    file_labels.close()

sys.argv.append(1)
if __name__=='__main__':
    if len(sys.argv) >1 :
        print('job index',sys.argv[1])
        index=int(sys.argv[1])
        pdm_factor=20
        mode=white_list_mode[index]
        #setupPDMtoText(pdm_factor=pdm_factor,mode=mode)

        print('testing')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = SubsetPDM().to(device)
        print(a[1])


