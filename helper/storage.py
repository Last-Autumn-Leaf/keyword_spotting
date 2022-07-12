from helper.utilsFunc import currentOrLast


class Storage:
    def __init__(self,exp_name='nameless_exp'):
        self.storage={'exp_name':exp_name}
        self.exp_index=0
        self.model=[]
        self.batch_size=[]
        self.num_epochs=[]

        # OPTIM PARAMS
        self.lr=[]
        self.weight_decay=[]

        # SCHEDULER PARAMS
        self.step_size=[]
        self.gamma=[]

        # MODEL PARAMS


    def __getitem__(self, item):
        return self.storage[item]

    def __setitem__(self, key, value):
        self.storage[key]=value



    def step(self):
        self.exp_index+=1



if __name__=='__main__':

    a=Storage()


    print('a')