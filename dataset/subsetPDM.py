import sys

import torch
import os
import pickle

from torch.utils.data import Dataset

from dataset.subsetSC import SubsetSC
from helper.utilsFunc import PdmTransform


class SubsetPDM(Dataset):
    def __init__(self):
        ...

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

    def __len__(self): return len(self.target)
    def __getitem__(self, i): return (self.data[i],self.target[i])

white_list_mode=['training','testing','validation']
def setupPDM(pdm_factor=20,mode='training'):
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
    PDMsubset = SubsetPDM()
    label,waveform=[],[]
    for i in range(n):
        temp=subset[i]
        label.append(temp[2])
        waveform.append(temp[0])

    # padding :
    print('padding...')
    waveform = subset.pad_sequence(waveform)
    # Resample et PDM transform
    print('PDM-transforming...')
    waveform = PDM_TRAMSFORM(waveform)

    #saving
    print('dumping')
    PDMsubset.setData(waveform)
    PDMsubset.setTarget(label)
    PDMsubset.getLabels()
    pickle.dump(PDMsubset, open("PDM_dataset_"+mode+'_'+str(pdm_factor)+".pt", "wb"))


if __name__=='__main__':
    if len(sys.argv) >1 :
        print('job index',sys.argv[1])
        index=int(sys.argv[1])
        pdm_factor=20
        mode=white_list_mode[index]
        setupPDM(pdm_factor=pdm_factor,mode=mode)

        print('testing')
        PDMsubset = pickle.load(open("PDM_dataset_"+mode+'_'+str(pdm_factor)+".pt", "rb"))
        a=PDMsubset[0]
        b=PDMsubset[0:9]
        print(a)
        print(b)
        print(len(PDMsubset))


