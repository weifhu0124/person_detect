import numpy as np
import glob
from PIL import Image

# read in data from folders path
# label is either 0 or 1
def read_data(path, label):
    data = []
    # get images from the directory
    images = glob.glob(path + '/*.bmp')
    for im in images:
        data.append((np.array(Image.open(im)), label))
    return np.array(data)

# shuffle data such that positive and negative examples are mixed
def shuffle_date(pos_data, neg_data):
    data = np.concatenate((pos_data, neg_data))
    return np.random.shuffle(data)

# test
if __name__ == '__main__':
    pos = read_data('data/bikes_and_persons', 1)
    neg = read_data('data/no_bike_no_person', 0)
    shuffle_date(pos, neg)


