from dataset.subsetSC import SubsetSC
import sys
import pickle
from .. import utilsFunc

def test_importing_dataset(root=None):
    print("testing importing dataset")

    with utilsFunc.timeThat('Test sets'):
        if root is None :
            root='./'
        test_set = SubsetSC("testing", root)
        print('test loader set up, size', len(test_set))
    print('import finished')

if __name__=='__main__':
    '''if len(sys.argv) >1 :
        print('root set to',sys.argv[1])
        r=sys.argv[1]
    else :
        r=None

    test_importing_dataset(r)'''


    root = './'
    with utilsFunc.timeThat('init dataset') :
        train_set = SubsetSC("training",root=root)
        validation_set = SubsetSC("validation",root=root)
        test_set = SubsetSC("testing",root=root)

    # do the savings :
    #Important notice: The maximum file size of pickle is about 2GB.
    #The advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster.
    with utilsFunc.timeThat('saving dataset'):
        with open('./SpeechCommands/pickle/test_set.pt', 'wb') as handle:
            pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./SpeechCommands/pickle/train_set.pt', 'wb') as handle:
            pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./SpeechCommands/pickle/validation_set.pt', 'wb') as handle:
            pickle.dump(validation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with utilsFunc.timeThat('reimporting dataset'):
        with open('./SpeechCommands/pickle/test_set.pt', 'rb') as handle:
            pickled_test_set = pickle.load(handle)
            waveform, sample_rate, label, speaker_id, utterance_number = pickled_test_set[0]
            print("test set imported")
        with open('./SpeechCommands/pickle/train_set.pt', 'rb') as handle:
            pickled_train_set = pickle.load(handle)
            waveform, sample_rate, label, speaker_id, utterance_number = pickled_train_set[0]
            print("train set imported")
        with open('./SpeechCommands/pickle/validation_set.pt', 'rb') as handle:
            pickled_validation_set = pickle.load(handle)
            waveform, sample_rate, label, speaker_id, utterance_number = pickled_validation_set[0]
            print("validation set imported")

    print("import successfull")