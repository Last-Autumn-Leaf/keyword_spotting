from helper.utilsFunc import currentOrLast


class Storage:
    def __init__(self,exp_name='nameless_exp',exp_index=0):
        self.data={'exp_name':exp_name,
                   'exp_index':exp_index}


    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key]=value






if __name__=='__main__':

    a=Storage()


    print('a')