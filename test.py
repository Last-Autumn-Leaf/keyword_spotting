import sys
import torch
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
def test():
    with timeThat('training/validation/tests sets' if not test_only else 'test set'):
        if not test_only :
            train_set = SubsetSC("training",'./')
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=b_size,
                shuffle=True,
                collate_fn=train_set.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory, )
            val_set = SubsetSC("validation", './')
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=b_size,
                shuffle=True,
                collate_fn=train_set.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory, )
        test_set = SubsetSC("testing", './')
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=b_size,
            shuffle=True,
            collate_fn=test_set.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory, )

    print('batch size :',b_size)
    with timeThat('Every batches'):
        for batch_idx, (data, target) in enumerate(test_loader):
            with timeThat('\tbatch '+str(batch_idx)+'/'+str(len(test_loader))):
                data = data.to(device)
                data = PDM_TRAMSFORM(data)

if __name__=='__main__':
    test()

    if len(sys.argv) >1 :
        print('job index',sys.argv[1])
