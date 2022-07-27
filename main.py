import torch.optim as optim
from tqdm import tqdm
import metrics.metrics as metrics
from dataset.subsetSC import SubsetSC
#import models
from helper.storage import Storage
from models.PDM_model import PDM_model
from models.spectrogram_model import spectrogram_model
from models.mel_model import *
from models.M5 import *

from helper.utilsFunc import *
import argparse
from os import makedirs
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

print( 'We are the',datetime.today().strftime('%d-%m-%Y %H:%M:%S') )
print('imports done')

new_sample_rate=8000
pdm_factor=10

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with
        datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage="\n python3 main.py [model] [hyper_parameters]"
                                           "\n python3 main.py --model PDM_model [hyper_parameters]"
                                            "\n python3 main.py [model] --predict [load_checkpoint]",
                                     description="This program allows to train different models of classification.")

    parser.add_argument("--exp_name", type=str,default="nameless_exp",
                        help="Name of experiment")
    parser.add_argument("--exp_index", type=int, default=0,
                        help="index of experiment")
    parser.add_argument("--model", type=str, default=spect_model,
                        choices=[M5_model, spect_model, MFCC_MODEL, PDM_MODEL, spect_MEL])
    parser.add_argument("--batch_size",   type=int, default=100,
                        help="The size of the training batch.")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument("--num-epochs",   type=int, default=2,
                        help="The number of epochs.")

    # OPTIM PARAMS
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",   type=float, default=0.0001,
                        help="decay of the weight.")
    # SCHEDULER PARAMS
    parser.add_argument("--step_size",   type=int, default=20,
                        help="step size of scheduler. ")
    parser.add_argument("--gamma",   type=float, default=0.1,
                        help="gamma of scheduler. ")

    # MODEL PARAMS

    # stride,n_channel,kernel_size,dilation
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n_channel", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=80)
    parser.add_argument("--dilation", type=int, default=1)


    parser.add_argument("--pdm_factor", type=int, default=pdm_factor,
                        help="pdm factor when using PDM model, by default set to 48 ")


    parser.add_argument("--predict", action="store_true",
                        help="Load weights to predict the mask of a randomly selected image from the test set")
    parser.add_argument("--log_interval", type=int, default=200,
                        help="The log interval of the training batch.")
    parser.add_argument("--load_checkpoint",   type=str,
                        help="Location of a training checkpoint to load")
    parser.add_argument("--nosavemodel", action="store_true",
                        help="don't save the model")
    parser.add_argument("--COLAB", action="store_true",
                        help="Change the root for storing the dataset")
    parser.add_argument("--no_validation", action="store_true",
                        help="Will not do the validation")

    parser.add_argument("--fe", type=int, default=new_sample_rate,
                        help="Sampling frequency in Hz")
    parser.add_argument("--n_mels", type=int, default=50,
                        help="numbers of mel filters")
    # This should be put LAST on the parser
    parser.add_argument("--win_length", type=int, default=int(30e-3 * new_sample_rate),
                        help="Window length")
    parser.add_argument("--hop_length", type=int, default=int(10e-3 * new_sample_rate),
                        help="hop length")
    parser.add_argument("--n_fft", type=int, default=int(30e-3 * new_sample_rate),
                        help="n_fft")

    
    return parser

def main(args):

    # We use a dictionnary-like data type to stored usefull variables.
    print('setting up exp:', args.exp_name)
    storage = Storage(args.exp_name,args.exp_index)
    storage['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using :', storage['device'])

    #Mode :
    storage['predict']=args.predict
    storage['no_validation']=args.no_validation
    storage['nosavemodel']=args.nosavemodel

    #H-PARAM
    storage['batch_size'] = args.batch_size
    storage['lr'] = args.lr
    storage['weight_decay'] = args.weight_decay
    storage['step_size'] = args.step_size
    storage['gamma'] = args.gamma

    #Downloading the DATASET
    root = '/content/sample_data' if args.COLAB else './'
    if 'SLURM_TMPDIR' in os.environ :
        root=os.environ['SLURM_TMPDIR']
        print('Compute Canada detected, root set to',root)


    if storage['device'] == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if not storage['predict']: # TRAINING AND OR VALIDATION MODE
        print('Training mode')
        with timeThat('training/validation sets'):
            train_set = SubsetSC("training", root)
            storage['train_loader'] = torch.utils.data.DataLoader(
                train_set,
                batch_size=storage['batch_size'],
                shuffle=True,
                collate_fn=train_set.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,)
            print('training loader set up, size',len(train_set))
            if not storage['no_validation'] :
                val_set = SubsetSC("validation", root)
                storage['val_loader'] = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=storage['batch_size'],
                    shuffle=True,
                    collate_fn=train_set.collate_fn,
                    num_workers=num_workers,
                    pin_memory=pin_memory, )
                print('validation loader set up, size',len(val_set))
            storage['waveform'], storage['sample_rate'], label, speaker_id, utterance_number = train_set[0]
    else : # PREDICTION MODE
        print('Prediction mode')
        with timeThat('test sets'):
            test_set = SubsetSC("testing", root)
            storage['test_loader'] = torch.utils.data.DataLoader(
                test_set,
                batch_size=storage['batch_size'],
                shuffle=True,
                collate_fn=test_set.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory, )
            print('test loader set up, size', len(test_set))
            storage['waveform'], storage['sample_rate'], label, speaker_id, utterance_number = test_set[0]

    storage['waveform']=storage['waveform'].to(storage['device'])



    #1D transform param
    storage['fe']=args.fe

    #models params
    storage['stride']=args.stride
    storage['n_channel']=args.n_channel
    storage['kernel_size']=args.kernel_size
    storage['dilation']=args.dilation



    # 2D transform params
    storage['n_mels'] = args.n_mels
    storage['win_length'] = args.win_length
    storage['hop_length'] = args.hop_length
    storage['n_fft'] = args.n_fft
    storage['pdm_factor']=args.pdm_factor

    # setting up the transforms and model
    storage['model_name'] = args.model
    storage['base_name'] =storage['model_name'] + '/' + storage['exp_name'] + '/' + str(storage['exp_index'])

    # TensorBoards :
    storage['writer'] = SummaryWriter('runs/' + storage['exp_name'] + '/' + str(storage['exp_index']),
                                      filename_suffix=storage['base_name'].replace('/','_'))
    print("saving tensorboard in runs/" + storage['exp_name'])
    
    if  storage['model_name'] == M5_model :
        storage['transform'] = torchaudio.transforms.Resample(orig_freq=storage['sample_rate'], new_freq=storage['fe']).to(storage['device'])
        
        # setting up the model
        waveform_size = storage['transform'](storage['waveform']).shape
        storage['model']= M5( n_output=len(test_set.labels if storage['predict'] else train_set.labels)).to(storage['device'])
        
        print('M5 model setup')
    elif  storage['model_name'] == spect_model:
        storage['transform']=torchaudio.transforms.Spectrogram(n_fft=storage['n_fft'],win_length=storage['win_length'],
                                                               hop_length= storage['hop_length'] ).to(storage['device'])

        waveform_size = storage['transform'](storage['waveform']).shape
        storage['model'] =spectrogram_model(input_shape=waveform_size,
                                            n_output=len(train_set.labels if not storage['predict'] else test_set.labels)).to(storage['device'])
        
        print('spect model setup')
    elif  storage['model_name'] == spect_MEL :
        storage['transform'] = torchaudio.transforms.MelSpectrogram(n_fft=storage['n_fft'],
                                                                         n_mels=storage['n_mels'], win_length=storage['win_length'],
                                                                         hop_length=storage['hop_length']).to(storage['device'])

        waveform_size = storage['transform'](storage['waveform']).shape
        storage['model']=mel_model(input_shape=waveform_size,
                                          n_output=len(train_set.labels if not storage['predict'] else test_set.labels)).to(storage['device'])
        
        print('MEL spectrogram model setup')
    elif storage['model_name'] == MFCC_MODEL :
        storage['transform'] = torchaudio.transforms.MFCC(melkwargs={
            "n_fft": storage['n_fft'],
            "n_mels": storage['n_mels'],
            "hop_length": storage['hop_length'],
            "mel_scale": "htk",
        }).to(storage['device'])
        
        waveform_size = storage['transform'](storage['waveform']).shape
        storage['model']=mel_model(input_shape=waveform_size, n_output=len(train_set.labels if not storage['predict'] else test_set.labels)).to(storage['device'])
        
        print('MFCC model setup')
    elif storage['model_name'] == PDM_MODEL :
        storage['transform']=PdmTransform(pdm_factor=storage['pdm_factor'],signal_len=len(storage['waveform'][0]),
                                   orig_freq=storage['sample_rate']).to(storage['device'])

        waveform_size = storage['transform'](storage['waveform']).shape
        storage['model']=PDM_model( n_output=len(test_set.labels if storage['predict'] else train_set.labels)) .to(storage['device'],
            stride=storage['stride'],n_channel=storage['n_channel'],kernel_size=storage['kernel_size'],dilation=storage['dilation'])
        
        print('PDM model setup')

    else :
        raise Exception(storage['model_name'] +" not implemented")

    # Storing the input form :
    storage['input']= storage['transform'](storage['waveform'])
    if storage['exp_index'] ==0 :
        storeInputForm(storage)


    
    # Define the optimizer, loss function & metrics
    
    storage['optimizer']=optim.Adam(storage['model'].parameters(), lr=storage['lr'], weight_decay=storage['weight_decay'])
    if not storage['predict'] :
        storage['scheduler']= optim.lr_scheduler.StepLR(storage['optimizer'], step_size=storage['step_size'],
                                    gamma=storage['gamma']) 
        
    
    # Define the loss Function
    storage['lossFunc'] = F.cross_entropy
    # Define the metrics :
    storage['metrics'] = metrics.countCorrectOutput

    # Define the log interval and epochs
    storage['log_interval'] = args.log_interval
    storage['num_epochs'] = args.num_epochs

    # loading the saved model
    print('Launching exp:', storage['exp_name'], 'index',storage['exp_index'])
    if args.load_checkpoint :
        storage['model'].load_state_dict( torch.load( storage['exp_name'] + '/' + args.load_checkpoint,map_location=storage['device'] ) )

    if not storage['predict'] :
        if not storage['nosavemodel']:
            makedirs('./saved_models/'+storage['model_name'] + '/' + storage['exp_name'], exist_ok=True)
        if storage['no_validation'] :
            storage['pbar_update'] = np.round(1 / len(storage['train_loader']),3)
        else :
            storage['pbar_update'] = np.round(1 / ( len(storage['train_loader']) + len(storage['val_loader']) ) ,3)
    else :
        storage['pbar_update'] = np.round(1 / (len(storage['test_loader'])), 3)


    with timeThat('main program',storage):
        with tqdm(total=storage['num_epochs']) as pbar:
            if not storage['predict']:

                storage['best_accuracy'] = 0
                storage['pbar'] = pbar
                storage['train_index'] = 0
                storage['val_index'] = 0
                storage['acc_index'] = 0
                for epoch in range(1, storage['num_epochs']+ 1):
                    storage['current_epoch'] = epoch

                    train(storage)
                    if not storage['no_validation']:
                        train(storage,validation=True)
                        storage['scheduler'].step()

                    # SAVE THE MODEL accuracy-wise
                    if not storage['nosavemodel'] and storage['best_accuracy'] < storage['accuracy']:
                        namefile='./saved_models/'+storage['model_name']+'/'+storage['exp_name']+'/'+str(storage['exp_index'])+'.pt'
                        torch.save(storage['model'].state_dict(),namefile)
                        print('saving model at',namefile, 'with accuracy=',storage['accuracy'])
                        storage['best_accuracy'] = storage['accuracy']

            else :
                test(storage)

            storage['writer'].flush()
    if not storage['predict']:
        storeWeights(storage)
        storeFeatureMaps(storage)
        storage.save_hparams()

    storage['writer'].close()

def storeWeights(storage,epoch=0):
    if storage['input'].ndim == 3:
        PlotKernelFunc = plot_kernels2D
    elif storage['input'].ndim == 2:
        PlotKernelFunc = plot_kernels1D

    # This supposed that every model got their first layer named conv1
    print('storing the weights')
    FirstLayerWeights = storage['model'].conv1.weight.detach().cpu().numpy()
    fig = PlotKernelFunc(FirstLayerWeights)
    name = storage['base_name'] + '/Weights'
    storage['writer'].add_figure(name, fig,epoch)

def storeFeatureMaps(storage,epoch=0):
    if storage['input'].ndim == 3:
        PlotKernelFunc = plot_kernels2D
    elif storage['input'].ndim == 2:
        PlotKernelFunc = plot_kernels1D

    print('storing the Feature maps')
    input=storage['input'].detach().cpu()
    FirstLayerWeights = storage['model'].conv1.weight.detach().cpu().numpy()

    featureMap = FirstLayerWeights( input )[:, None].detach().cpu().numpy()
    fig = PlotKernelFunc(featureMap)
    name = storage['base_name'] + '/FeatureMaps'
    storage['writer'].add_figure(name, fig,epoch)

def storeInputForm(storage):
    fig = plt.figure()
    if storage['input'].ndim == 3:
        plt.imshow(storage['input'].log2()[0].detach().cpu().numpy())
    elif storage['input'].ndim == 2:
        plt.plot(storage['input'].t().cpu().numpy())
    else:
        raise Exception('error on the shape of the input' + str(storage['input'].shape))
    name = storage['model_name'] + '/' + storage['exp_name'] + '/Input'
    storage['writer'].add_figure(name, fig)


if __name__ == '__main__':
    args = argument_parser()
    main(args.parse_args())
