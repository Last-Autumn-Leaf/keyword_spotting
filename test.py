import os
import pathlib
import sys
import zipfile

from dataset.subsetPDM import white_list_mode, setupPDMtoText, SubsetPDM
from helper.utilsFunc import timeThat
import torch

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__=='__main__':
    if len(sys.argv) > 1:
        print('job index', sys.argv[1])
        index = int(sys.argv[1])
        pdm_factor = 20
        mode = white_list_mode[index]
        root=pathlib.Path('./')
        if 'SLURM_TMPDIR' in os.environ:
            root = pathlib.Path(os.environ['SLURM_TMPDIR'])
            print('Compute Canada detected, root set to', root)

        with timeThat(mode +' PDM DATASET'):
            file_tensor,file_labels=setupPDMtoText(pdm_factor=pdm_factor,mode=mode,root=root,device=device)

        print('testing ...')
        a = SubsetPDM(mode=mode,root=root).to(device)
        print(a[1])
        print(len(a))

        # We do this only if we are on compute Canada
        if 'SLURM_TMPDIR' in os.environ:
            zip_path=pathlib.Path.cwd() / 'PDM_{}_{}.zip'.format(pdm_factor,mode)
            print('zipping files at :', zip_path)
            #zipping and sending it to the right folder
            with zipfile.ZipFile(zip_path, 'w',
                                 compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as zf:
                zf.write(file_tensor)
                zf.write(file_labels)


