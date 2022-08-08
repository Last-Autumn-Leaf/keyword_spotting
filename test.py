import pickle
import sys
import torch

from dataset.subsetPDM import white_list_mode, setupPDMtoText, SubsetPDM
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
        # setupPDMtoText(pdm_factor=pdm_factor,mode=mode)

        print('testing')
        a = SubsetPDM(mode=mode)
        print(a[1])
