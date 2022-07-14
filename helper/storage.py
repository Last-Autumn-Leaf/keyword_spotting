from helper.utilsFunc import *


class Storage:
    hprams=['lr','batch_size','num_epochs','weight_decay','step_size','gamma']
    hparams_PDM=['pdm_factor']
    hparams_spec=['n_fft','hop_length','win_length']
    hparams_MEL=hparams_spec.copy()
    hparams_MEL.append('n_mels')
    hparams_MFCC=hparams_MEL.copy()
    hparams_MFCC.remove('win_length')
    hparams_M5=['fe']
    def __init__(self,exp_name='nameless_exp',exp_index=0):
        self.data={'exp_name':exp_name,
                   'exp_index':exp_index}


    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key]=value

    def save_hparams(self):
        if self.data['model_name'] ==M5_model :
            true_param=self.hprams +self.hparams_M5
        elif    self.data['model_name'] ==spect_model:
            true_param =self.hprams +self.hparams_spec
        elif    self.data['model_name'] ==MFCC_MODEL:
            true_param =self.hprams +self.hparams_MFCC
        elif    self.data['model_name'] ==PDM_MODEL:
            true_param =self.hprams +self.hparams_PDM
        elif    self.data['model_name'] ==spect_MEL:
            true_param =self.hprams +self.hparams_MEL
        else :
            raise Exception('hparams for model '+self.data['model_name']+' not found !')

        hparam_dict={ key:self.data[key] for key in true_param}
        metric_dict={'accuracy':self.data['best_accuracy']}

        self.data['writer'].add_hparams(
            hparam_dict,metric_dict,
            run_name=self.data['base_name']
        )



if __name__=='__main__':

    a=Storage()


    print('a')