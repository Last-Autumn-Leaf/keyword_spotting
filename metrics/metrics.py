
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def countCorrectOutput(output,target):
    pred=output.argmax(dim=-1)
    return pred.squeeze().eq(target).sum().item()


#TODO : confusion matrix