
import random
import sys

import main
import helper.utilsFunc
from models.PDM_model import try_param

MAX_LR=0.001
MIN_LR =0.000001
MAX_weight_decay=MAX_LR/10
MIN_weight_decay=MIN_LR/10

DILATION = (1,50)
KERNEL=(100,10000)
STRIDE=(4,50)

#random.randint() can be
def createParams():
    return {
        '--exp_name': 'test_PDM',
        '--model': helper.utilsFunc.PDM_MODEL,
        '--num-epochs': 50,
        '--pdm_factor': 10,
        '--lr': MAX_LR / 100,
        '--weight_decay': MAX_LR / 1000,

        '--stride': random.randint(STRIDE[0], STRIDE[1]),
        '--n_channel': 20,
        '--kernel_size': random.randint(KERNEL[0], KERNEL[1]),
        '--dilation': random.randint(DILATION[0], DILATION[1])
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

    a=try_param(pdm_factor,stride,n_channel,kernel_size,dilation)
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
    return args_list

if __name__ == '__main__':
    if len(sys.argv) > 1:
        current_index=int(sys.argv[1])
        sys.argv = [sys.argv[0]]
        args = main.argument_parser()

        if validate() :
            args=args.parse_args(deploy(current_index))
            print('args: ',args)
            main.main(args)

