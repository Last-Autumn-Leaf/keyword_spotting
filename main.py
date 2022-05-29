
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

import metrics.metrics as metrics
from dataset.subsetSC import SubsetSC, resample
from models.M5 import *

from utilsFunc import *


def main():


    '''def predict(tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = transform(tensor)
        tensor = model(tensor.unsqueeze(0))
        tensor = metrics.get_likely_index(tensor)
        tensor = metrics.index_to_label(tensor.squeeze())
        return tensor'''

    storage=dict()

    storage['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using :',storage['device'])

    # Create training and testing split of the data. We do not use validation in this tutorial.
    with timeThat('training & test sets'):
        train_set = SubsetSC("training")
        test_set = SubsetSC("testing")

    storage['waveform'], storage['sample_rate'], label, speaker_id, utterance_number = train_set[0]

    #Resampling :
    waveform_size=resample(storage,new_sample_rate=8000)
    batch_size = 256

    if storage['device'] == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # setting up the LOADER
    storage['train_loader']  = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    storage['test_loader']  = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=test_set.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Define the NETWORK
    storage['model'] = M5(n_input=waveform_size, n_output=len(train_set.labels))
    storage['model'].to(storage['device'])
    print(storage['model'])
    print("Number of parameters: %s" % storage['model'].count_parameters())

    # Define the Optimizer
    storage['optimizer'] = optim.Adam(storage['model'].parameters(), lr=0.01, weight_decay=0.0001)
    storage['scheduler'] = optim.lr_scheduler.StepLR(storage['optimizer'], step_size=20,
                                          gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

    #Define the loss Function
    storage['lossFunc']=F.nll_loss

    #Define the metrics :
    storage['metrics']=metrics.countCorrectOutput

    storage['log_interval'] = 20
    storage['n_epoch'] = 2

    storage['pbar_update']  = 1 / (len(storage['train_loader']) + len(storage['test_loader']))
    storage['losses'] = []

    # The transform needs to live on the same device as the model and the data.
    storage['transform'] = storage['transform'].to(storage['device'])
    with timeThat('Main program') :
        with tqdm(total=storage['n_epoch']) as pbar:
            storage['pbar']=pbar
            for epoch in range(1, storage['n_epoch'] + 1):
                train(storage)
                test(storage)
                storage['scheduler'].step()

    plt.plot(storage['losses']);
    plt.title("training loss");
    showResult(storage['model'])

if __name__ == '__main__':

    main()
