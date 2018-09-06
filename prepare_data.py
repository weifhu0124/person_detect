import numpy as np
import glob
import os
from PIL import Image

# corp photos into 480x480
def preprocess(path, save_path):
    H = 480
    W = 480
    images = glob.glob(path + '/*.bmp')
    # create new processed folder
    if not os.path.isdir('data/processed/'):
        os.mkdir('data/processed/')
    # make a new directory if not exist
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for im in images:
        image = Image.open(im)
        image = image.resize((W, H), Image.ANTIALIAS)
        image.save(save_path + os.path.basename(im))

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
def shuffle_data(pos_data, neg_data):
    data = np.concatenate((pos_data, neg_data))
    return np.random.shuffle(data)

# test
if __name__ == '__main__':
    positve_processed = 'data/processed/positive/'
    negative_processed = 'data/processed/negative/'
    preprocess('data/bikes_and_persons', positve_processed)
    preprocess('data/no_bike_no_person', negative_processed)


