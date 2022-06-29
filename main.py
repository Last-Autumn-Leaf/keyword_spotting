import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio,torchvision
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
import metrics.metrics as metrics
from dataset.subsetSC import SubsetSC
#import models
from models.spectrogram_model import spectrogram_model
from models.mel_model import *
from models.M5 import *

from utilsFunc import *
import argparse
from os import makedirs
import os
import pickle

print('imports done')
spect_model='spect'
M5_model='M5'
MEL_MODEL='mel'
PDM_MODEL='PDM'
new_sample_rate=8000
pdm_factor=10

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with
        datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage="\n python3 main.py [model] [hyper_parameters]"
                                           "\n python3 main.py --model M5 [hyper_parameters]"
                                            "\n python3 main.py [model] --predict [load_checkpoint]",
                                     description="This program allows to train different models of classification.")

    parser.add_argument("--exp_name", type=str,default="nameless_exp",
                        help="Name of experiment")
    parser.add_argument("--model", type=str, nargs="+", default=spect_model,
                        choices=[M5_model, spect_model, MEL_MODEL,PDM_MODEL])
    parser.add_argument("--batch_size", nargs="+", type=int, default=100,
                        help="The size of the training batch. Accepts multiple space-seperated values for hyperparameter search (--batch_size 5 10 20)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument("--num-epochs", nargs="+", type=int, default=2,
                        help="The number of epochs. Accepts multiple space-seperated values for hyperparameter search (--num-epochs 10 15 20)")

    # OPTIM PARAMS
    parser.add_argument("--lr", nargs="+", type=float, default=0.001,
                        help="Learning rate. Accepts multiple space-seperated values for hyperparameter search (--lr 0.001 0.01 0.1)")
    parser.add_argument("--weight_decay", nargs="+", type=float, default=0.0001,
                        help="decay of the weight. Accepts multiple space-seperated values for hyperparameter search (--weight_decay 0.001 0.01 0.1)")
    # SCHEDULER PARAMS
    parser.add_argument("--step_size", nargs="+", type=int, default=20,
                        help="step size of scheduler. Accepts multiple space-seperated values for hyperparameter search (--step_size 0.001 0.01 0.1)")
    parser.add_argument("--gamma", nargs="+", type=float, default=0.1,
                        help="gamma of scheduler. Accepts multiple space-seperated values for hyperparameter search (--gamma 0.001 0.01 0.1)")

    # MODEL PARAMS
    parser.add_argument("--pdm_factor", type=int, default=pdm_factor,
                        help="pdm factor when using PDM model, by default set to 48 ")

    parser.add_argument("--predict", action="store_true",
                        help="Load weights to predict the mask of a randomly selected image from the test set")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="The log interval of the training batch. Accepts multiple space-seperated values for hyperparameter search (--log_interval 5 10 20)")
    parser.add_argument("--load_checkpoint", nargs="+", type=str,
                        help="Location of a training checkpoint to load")
    parser.add_argument("--save_checkpoint", nargs="+", type=str,
                        help="Location of a training checkpoint to save")
    parser.add_argument("--COLAB", action="store_true",
                        help="Change the root for storing the dataset")
    parser.add_argument("--no_validation", action="store_true",
                        help="Will not do the validation")

    parser.add_argument("--noshowplot", action="store_true",
                        help="don't show plot")
    parser.add_argument("--nosaveplot", action="store_true",
                        help="don't save plot")

    parser.add_argument("--without_pickled_data", action="store_true",
                        help="don't use pickle to load dataset")

    parser.add_argument("--fe", type=int, default=16000,
                        help="Sampling frequency in Hz")
    parser.add_argument("--n_mels", type=int, default=50,
                        help="numbers of mel filters")
    # This should be put LAST on the parser
    parser.add_argument("--win_length", type=int, default=int(30e-3 * parser.parse_args().fe),
                        help="Window length")
    parser.add_argument("--hop_length", type=int, default=int(10e-3 * parser.parse_args().fe),
                        help="hop length")
    parser.add_argument("--n_fft", type=int, default=parser.parse_args().win_length,
                        help="n_fft")

    
    return parser.parse_args()


def main():
    print('parsing variables...')
    args = argument_parser()
    if isinstance( args.model,str) :
        args.model=[args.model]
    if isinstance( args.batch_size,int) :
        args.batch_size=[args.batch_size]
    if isinstance( args.num_epochs,int) :
        args.num_epochs=[args.num_epochs]
    if isinstance( args.lr,float) :
        args.lr=[args.lr]
    if isinstance( args.weight_decay,float) :
        args.weight_decay=[args.weight_decay]
    if isinstance( args.step_size,int) :
        args.step_size=[args.step_size]
    if isinstance( args.gamma,float) :
        args.gamma=[args.gamma]


    if isinstance(args.load_checkpoint,str) :
        args.load_checkpoint=[args.load_checkpoint]
    if isinstance(args.save_checkpoint,str) :
        args.save_checkpoint=[args.save_checkpoint]



    # We use a dictionnary to stored usefull variables.
    storage = dict()
    storage['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using :', storage['device'])


    #Downloading the DATASET
    root = '/content/sample_data' if args.COLAB else './'
    if 'SLURM_TMPDIR' in os.environ :
        root=os.environ['SLURM_TMPDIR']
        print('Compute Canada detected, root set to',root)
        print('We will not use pickled data')
        args.without_pickled_data = True

    if storage['device'] == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if not args.predict: # TRAINING AND OR VALIDATION MODE
        print('Training mode')
        with timeThat('training/validation sets'):
            if 'train_set' not in locals():
                if args.without_pickled_data :
                    train_set = SubsetSC("training", root)
                else :
                    with open('./SpeechCommands/pickle/train_set.pt', 'rb') as handle:
                        train_set = pickle.load(handle)
                storage['train_loader']=[]
                for batch_size in args.batch_size :
                    storage['train_loader'] .append( torch.utils.data.DataLoader(
                        train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=train_set.collate_fn,
                        num_workers=num_workers,
                        pin_memory=pin_memory,) )
                print('training loader set up, size',len(train_set))
            if not args.no_validation :
                if 'val_set' not in locals():
                    if args.without_pickled_data:
                        val_set = SubsetSC("validation", root)
                    else :
                        with open('./SpeechCommands/pickle/validation_set.pt', 'rb') as handle:
                            val_set = pickle.load(handle)
                    storage['val_loader']=[]
                    for batch_size in args.batch_size:
                        storage['val_loader'].append(torch.utils.data.DataLoader(
                            val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=train_set.collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory, ))
                    print('validation loader set up, size',len(val_set))
            storage['waveform'], storage['sample_rate'], label, speaker_id, utterance_number = train_set[0]
    else : # PREDICTION MODE
        print('Prediction mode')
        with timeThat('test sets'):
            if 'test_set' not in locals():
                if args.without_pickled_data:
                    test_set = SubsetSC("testing", root)
                else :
                    with open('./SpeechCommands/pickle/test_set.pt', 'rb') as handle:
                        test_set = pickle.load(handle)
                storage['test_loader'] = []
                for batch_size in args.batch_size:
                    storage['test_loader'].append(torch.utils.data.DataLoader(
                        test_set,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=test_set.collate_fn,
                        num_workers=num_workers,
                        pin_memory=pin_memory, ))
                print('test loader set up, size', len(test_set))
            storage['waveform'], storage['sample_rate'], label, speaker_id, utterance_number = test_set[0]

    storage['waveform']=storage['waveform'].to(storage['device'])


    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rates = args.lr

    # 2D transform params
    n_mels = args.n_mels
    win_length = args.win_length
    hop_length = args.hop_length
    n_fft = args.n_fft



    # setting up the transforms and model
    storage['transform']=[]
    storage['model']=[]
    c=0

    for model in args.model :

        if model == M5_model :
            M5_transform = torchaudio.transforms.Resample(orig_freq=storage['sample_rate'], new_freq=new_sample_rate)
            storage['transform'].append(M5_transform)
            # The transform needs to live on the same device as the model and the data.
            storage['transform'][-1] = storage['transform'][-1].to(storage['device'])
            waveform_size = storage['transform'][-1](storage['waveform']).shape
            storage['model'].append(M5( n_output=len(test_set.labels if args.predict else train_set.labels)) )
            storage['model'][-1].to(storage['device'])
            print('M5 model setup')
        elif model == spect_model:
            #setting up the correct transform
            storage['transform'].append(torchaudio.transforms.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length))
            # The transform needs to live on the same device as the model and the data.
            storage['transform'][-1] = storage['transform'][-1].to(storage['device'])
            # setting up the model
            waveform_size = storage['transform'][-1](storage['waveform']).shape
            storage['model'] .append( spectrogram_model(input_shape=waveform_size, debug=True,n_output=len(train_set.labels if not args.predict else test_set.labels)) )
            storage['model'][-1].to(storage['device'])
            print('spect model setup')
        elif model == MEL_MODEL :
            MFCC_transform = torchaudio.transforms.MFCC(melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            })
            storage['transform'].append(MFCC_transform)
            # The transform needs to live on the same device as the model and the data.
            storage['transform'][-1] = storage['transform'][-1].to(storage['device'])
            waveform_size = storage['transform'][-1](storage['waveform']).shape
            storage['model'].append(mel_model(input_shape=waveform_size, n_output=len(train_set.labels if not args.predict else test_set.labels)))
            storage['model'][-1].to(storage['device'])
            print('MEL model setup')
        elif model == PDM_MODEL :

            PDM_transform=PdmTransform(pdm_factor=args.pdm_factor,signal_len=len(storage['waveform'][0]),
                                       orig_freq=storage['sample_rate'])
            storage['transform'].append(PDM_transform)
            # The transform needs to live on the same device as the model and the data.

            storage['transform'][-1] = storage['transform'][-1].to(storage['device'])

            waveform_size = storage['transform'][-1](storage['waveform']).shape
            storage['model'].append(M5( n_output=len(test_set.labels if args.predict else train_set.labels)) )
            storage['model'][-1].to(storage['device'])
            print('PDM model setup')

        else :
            raise Exception(model +" not implemented")
        c+=1
    print(len(args.model),' models have been set up')


    # Define the optimizer, loss function & metrics
    storage['optimizer']=[]
    storage['scheduler']=[]
    for i in range(len(args.model)) :

        storage['optimizer'].append( optim.Adam(storage['model'][i].parameters(), lr=args.lr[currentOrLast(i,args.lr)], weight_decay=args.weight_decay[currentOrLast(i,args.weight_decay)]))
        if not args.predict :
            storage['scheduler'].append( optim.lr_scheduler.StepLR(storage['optimizer'][i], step_size=args.step_size[currentOrLast(i,args.step_size)],
                                        gamma=args.gamma[currentOrLast(i,args.gamma)]) ) # reduce the learning after 20 epochs by a factor of 10
    # TODO : maybe we can set this too in the parser
    # Define the loss Function
    storage['lossFunc'] = F.cross_entropy
    # Define the metrics :
    storage['metrics'] = metrics.countCorrectOutput

    # Define the log interval and epochs
    storage['log_interval'] = args.log_interval
    storage['n_epoch'] = args.num_epochs

    # loading the saved model
    print('Launching ', args.exp_name, 'experience')
    if args.load_checkpoint :
        for i in range(len(storage['model'])):
            storage['model'][i].load_state_dict( torch.load( args.exp_name + '/' + args.load_checkpoint[currentOrLast(i, args.load_checkpoint)],map_location=storage['device'] ) )

    if not args.predict :
        makedirs(args.exp_name, exist_ok=True)
        storage['losses_train'] = {i: [] for i in range(len(storage['model']))}
        if args.no_validation :
            storage['pbar_update'] = [np.round(1 / len(storage['train_loader'][i]),3) for i in range(len(args.batch_size))]

        else :
            storage['pbar_update'] = [ np.round(1 / ( len(storage['train_loader'][i]) + len(storage['val_loader'][i]) ) ,3)for i in range(len(args.batch_size))]
            storage['losses_val'] = {i: [] for i in range(len(storage['model']))}
    else :
        storage['pbar_update'] = [np.round(1 / (len(storage['test_loader'][i])), 3) for i in range(len(args.batch_size))]

    with timeThat('main program'):
        #TODO  : It's here that train the differents models
        for exp_i in range(len(storage['model'])):
            print('expérience :',exp_i)
            best_model_stats=0

            with tqdm(total=storage['n_epoch'][currentOrLast(exp_i,storage['n_epoch'])]) as pbar:
                storage['pbar'] = pbar
                for epoch in range(1, storage['n_epoch'][exp_i] + 1):
                    storage['epoch'] = epoch
                    if not args.predict :
                        train(storage,exp_i)
                        if not args.no_validation:
                            train(storage,exp_i,True)
                            storage['scheduler'][exp_i].step()
                        # Do the saving accuracy_wise
                        if args.save_checkpoint and best_model_stats < storage['accuracy']:
                            torch.save(storage['model'][exp_i].state_dict(),
                                       args.exp_name + '/' + args.save_checkpoint[
                                           currentOrLast(exp_i, args.save_checkpoint)] +'.pt')
                            print('saving model',args.exp_name + '/' + args.save_checkpoint[
                                           currentOrLast(exp_i, args.save_checkpoint)]+'.pt')
                    else :
                        test(storage,exp_i)

            # TODO : rajouter l'accuracy
            if not args.predict and (not args.nosaveplot  or args.noshowplot):
                plt.plot(storage['losses_train'][exp_i], label='train loss')
                if not args.no_validation:
                    plt.plot(storage['losses_val'][exp_i], label='validation loss')
                plt.title("training/val loss exp" + str(exp_i))
                plt.legend()

                if not args.nosaveplot  :
                    savename = args.save_checkpoint[currentOrLast(exp_i, args.save_checkpoint)] if args.save_checkpoint else \
                        args.model[exp_i]+'_'+str(exp_i)
                    plt.savefig(args.exp_name + '/' +savename+'.png' )
                    print('saving plot :',args.exp_name + '/' +savename+'.png' )
                if not args.noshowplot :
                    plt.show()



if __name__ == '__main__':
    main()
