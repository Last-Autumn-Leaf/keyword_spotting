
import random
import sys

import main
import helper.utilsFunc

MAX_LR=0.001
MIN_LR =0.000001
MAX_weight_decay=MAX_LR/10
MIN_weight_decay=MIN_LR/10

DILATION = (1,100)
KERNEL=(10,1000)
STRIDE=(20,1000)

#random.randint() can be
PDM_search_dict={
    '--exp_name' : 'test_PDM',
    '--model' : helper.utilsFunc.PDM_MODEL,
    '--num_epochs' : 50,
    '--PDM_factor ': 10,
    '--lr ': MAX_LR / 100,
    '--weight_decay ': MAX_LR / 1000,

    '--stride':random.randint(STRIDE[0],STRIDE[1]),
    '--n_channel':10,
    '--kernel_size':random.randint(KERNEL[0],KERNEL[1]),
    '--dilation':random.randint(DILATION[0],DILATION[1])
}

def deploy(current_index):
    args_list=[]
    for key in PDM_search_dict:
        args_list.append(key)
        value=PDM_search_dict[key]
        if  isinstance(value,list):
            args_list.append(value[helper.utilsFunc.currentOrLast(current_index, value)])
        else:
            args_list.append(value)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        current_index=int(sys.argv[1])
        sys.argv = [sys.argv[0]]
        args = main.argument_parser()
        args=args.parse_args(deploy(current_index))
        main.main(args)

