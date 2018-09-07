import prepare_data as pdata
import numpy as np

# one-hot encoding for classification error
def one_hot_encode(label):
    if label == 0:
        return [1,0]
    else:
        return [0,1]

# load shuffled data and label
def load_data_label(train):
    # load training data
    if train == True:
        # use prepare_data.py to load data
        positve_processed = 'data/processed/train/positive/'
        negative_processed = 'data/processed/train/negative/'
    else: # load validation data
        positve_processed = 'data/processed/val/positive/'
        negative_processed = 'data/processed/val/negative'
    pos_data = pdata.read_data(positve_processed, label=1)
    neg_data = pdata.read_data(negative_processed, label=0)
    # concatenate and shuffle positive and negative samples
    X_data_label = np.concatenate((pos_data, neg_data))
    np.random.shuffle(X_data_label)
    # separate data and label
    data = []
    label = []
    for data_label in X_data_label:
        data.append(data_label[0])
        label.append(one_hot_encode(data_label[1]))
    return np.array(data), np.array(label)

# test
if __name__ == '__main__':
    print(len(load_data_label(train=True)[0]))
    print(len(load_data_label(train=False)[0]))