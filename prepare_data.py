import numpy as np
import glob
import os
from PIL import Image

# corp photos into 48x48
def preprocess(path, save_path):
    H = 48
    W = 48
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

# separate training and validation data
def split_train_val(path, positive):
    # create new directories
    if not os.path.isdir('data/processed/train'):
        os.mkdir('data/processed/train')
    if not os.path.isdir('data/processed/val'):
        os.mkdir('data/processed/val')
    if positive == True:
        TRAIN = 'data/processed/train/positive/'
        VAL = 'data/processed/val/positive/'
    else:
        TRAIN = 'data/processed/train/negative/'
        VAL = 'data/processed/val/negative/'
    # create new directories
    if not os.path.isdir(TRAIN):
        os.mkdir(TRAIN)
    if not os.path.isdir(VAL):
        os.mkdir(VAL)
    images = glob.glob(path + '*.bmp')
    counter = 0
    for im in images:
        image = Image.open(im)
        if counter < 48:
            # put in validation set
            image.save(VAL + os.path.basename(im))
        else:
            image.save(TRAIN + os.path.basename(im))
        counter += 1

# run preprocess once
if __name__ == '__main__':
    positve_processed = 'data/processed/positive/'
    negative_processed = 'data/processed/negative/'
    #preprocess('data/bikes_and_persons', positve_processed)
    #preprocess('data/no_bike_no_person', negative_processed)
    split_train_val(positve_processed, positive=True)
    split_train_val(negative_processed, positive=False)


