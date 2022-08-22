
import random
import sys

import main
import helper.utilsFunc
from models.PDM_model import try_param

MAX_LR=0.001
MIN_LR =0.000001
MAX_weight_decay=MAX_LR/10
MIN_weight_decay=MIN_LR/10

pdm_factor=20
fe=16000
DILATION = 40
KERNEL=80

#STRIDE=(1, int(2 *fe*20/1000) ) # 2ms
STRIDE=1
n_channel=32


#random.randint() can be
def createParams():
    stride=STRIDE

    Lin=fe*pdm_factor
    Lout= int( (Lin-DILATION*(KERNEL-1) -1)/stride +1)
    maxpool= int( (Lout +1)/124)
    return {
        '--exp_name': 'True_PDM',
        '--model': helper.utilsFunc.PDM_MODEL,
        '--num-epochs': 70,
        '--pdm_factor': pdm_factor,
        '--lr': MAX_LR / 100,
        '--weight_decay': MAX_LR / 1000,

        '--stride': stride,
        '--n_channel': n_channel,
        '--kernel_size': KERNEL,
        '--dilation': DILATION,
        '--batch_size' : 2,
        '--maxpool':maxpool
    }

PDM_search_dict= createParams()
MAX_DEPTH=10

def validate(depth=0):
    global PDM_search_dict

    if depth > MAX_DEPTH:
        print('stopping the search :\n','validate max depth reached')
        return False

    depth+=1
    pdm_factor=PDM_search_dict['--pdm_factor']
    stride=PDM_search_dict['--stride']
    n_channel=PDM_search_dict['--n_channel']
    kernel_size=PDM_search_dict['--kernel_size']
    dilation=PDM_search_dict['--dilation']
    maxpool=PDM_search_dict['--maxpool']

    a=try_param(PDM_factor=pdm_factor,stride=stride,
                n_channel=n_channel,kernel_size=kernel_size,
                dilation=dilation,maxpool=maxpool)
    if a == False :
        #reset_dict
        PDM_search_dict = createParams()
        print('creating a new dict for params')
        return validate(depth)
    else :
        (stride, n_channel, kernel_size, dilation)=a
        PDM_search_dict['--pdm_factor']=pdm_factor
        PDM_search_dict['--stride']=stride
        PDM_search_dict['--n_channel']=n_channel
        PDM_search_dict['--kernel_size']=kernel_size
        PDM_search_dict['--dilation']=dilation
        PDM_search_dict['--maxpool']=maxpool

        return True

def deploy(current_index):
    args_list=[]
    for key in PDM_search_dict:
        args_list.append(key)
        value=PDM_search_dict[key]
        if  isinstance(value,list):
            args_list.append(str(value[helper.utilsFunc.currentOrLast(current_index, value)]))
        else:
            args_list.append(str(value))
    args_list.append("--exp_index")
    args_list.append(str(current_index))
    return args_list

if __name__ == '__main__':
    if len(sys.argv) > 1:
        current_index=int(sys.argv[1])
        sys.argv = [sys.argv[0]]
        args = main.argument_parser()

        args=args.parse_args( deploy(current_index) )
        print('args: ',args)
        main.main(args)

