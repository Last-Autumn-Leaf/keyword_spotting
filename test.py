from dataset.subsetSC import SubsetSC
from utilsFunc import timeThat
import sys

def test_importing_dataset(root=None):
    print("testing importing dataset")
    with timeThat('Test sets'):
        if root is None :
            root='./'
        test_set = SubsetSC("testing", root)
        print('test loader set up, size', len(test_set))
    print('import finished')

if __name__=='__main__':
    if len(sys.argv) >1 :
        print('root set to',sys.argv[1])
        r=sys.argv[1]
    else :
        r=None

    test_importing_dataset(r)