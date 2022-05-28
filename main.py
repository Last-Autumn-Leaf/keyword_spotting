
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

import metrics.metrics as metrics
from dataset.subsetSC import SubsetSC
from models.M5 import M5

from utilsFunc import *


def main():
    def train(model, epoch, log_interval):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = F.nll_loss(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # update progress bar
            pbar.update(pbar_update)
            # record loss
            losses.append(loss.item())

    def test(model, epoch):
        model.eval()
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = model(data)

            pred = metrics.get_likely_index(output)
            correct += metrics.number_of_correct(pred, target)

            # update progress bar
            pbar.update(pbar_update)

        print(
            f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

    def predict(tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = transform(tensor)
        tensor = model(tensor.unsqueeze(0))
        tensor = metrics.get_likely_index(tensor)
        tensor = metrics.index_to_label(tensor.squeeze())
        return tensor

    new_sample_rate = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using :',device)

    # Create training and testing split of the data. We do not use validation in this tutorial.
    with timeThat('training & test sets'):
        train_set = SubsetSC("training")
        test_set = SubsetSC("testing")
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

    #Resampling :
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)

    labels=test_set.labels

    batch_size = 256

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # setting up the LOADER
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=test_set.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Define the NETWORK
    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    model.to(device)
    print(model)
    print("Number of parameters: %s" % model.count_parameters())

    # Define the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                          gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

    log_interval = 20
    n_epoch = 50

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transform.to(device)
    with timeThat('Main program') :
        with tqdm(total=n_epoch) as pbar:
            for epoch in range(1, n_epoch + 1):
                train(model, epoch, log_interval)
                test(model, epoch)
                scheduler.step()

    plt.plot(losses);
    plt.title("training loss");
    showResult(model)

def showResult(model):
    FirstLayerWeights = model.conv1.weight.detach().numpy()
    plot_kernels1D(FirstLayerWeights)
    plot_kernels1D(FirstLayerWeights, True)
    SecondLayerWeights =model.conv2.weight.detach().numpy()
    plot_kernels1D(SecondLayerWeights)
    plot_kernels1D(SecondLayerWeights, True)



if __name__ == '__main__':

    main()
