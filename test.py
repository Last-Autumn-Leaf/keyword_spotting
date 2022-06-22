from dataset.subsetSC import SubsetSC
from utilsFunc import timeThat


def test_importing_dataset():
    print("testing importing dataset")
    with timeThat('Test sets'):
        root='./'
        test_set = SubsetSC("testing", root)
        print('test loader set up, size', len(test_set))
    print('import finished')

if __name__=='__main__':
    test_importing_dataset()