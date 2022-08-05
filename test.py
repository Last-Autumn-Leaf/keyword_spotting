import pickle
import sys
import torch

from dataset.subsetPDM import white_list_mode, setupPDM
from dataset.subsetSC import SubsetSC
from helper.utilsFunc import PdmTransform, timeThat

b_size=100
PDM_factor=20
fe=16000
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PDM_TRAMSFORM=PdmTransform(pdm_factor=PDM_factor,signal_len=fe,
                                   orig_freq=fe).to(device)
test_only=True

if device== "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


if __name__=='__main__':
    if len(sys.argv) > 1:
        print('job index', sys.argv[1])
        index = int(sys.argv[1])
        pdm_factor = 20
        mode = white_list_mode[index]
        setupPDM(pdm_factor=pdm_factor, mode=mode)

        PDMsubset = pickle.load(open("PDM_dataset_" + mode + '_' + str(pdm_factor) + ".pt", "rb"))
        a = PDMsubset[0]
        b = PDMsubset[0:9]
        print(a)
        print(b)
        print(len(PDMsubset))
