# Tensorflow Person Detection Side Project

I created a binary classifier that determines whether a photo contains persons and bikes

## Code Files:
prepare_data.py: data preprocessing- resize images, separate training and validation data  
util.py: utility functions  
model.py: convolutional neural network architecture  
train.py: training the cnn  
inference.py: run inference on data  

## Dataset:
data are collected from ImageNet Person - Pedestrain category: http://image-net.org/explore.php  
## Result:
currently with a learning rate of 5e-4 and 500 steps, the accuracy can acheive 0.854