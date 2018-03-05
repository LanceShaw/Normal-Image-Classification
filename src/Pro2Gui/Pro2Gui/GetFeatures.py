# this py file is used to get features value of image in train set and images in test set
# then get the top10 knn neighbors for every query and output them in a txt file

import h5py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from get_data_of_train import get_test_from_dir
from get_data_of_train import get_testpic_from_dir
from sklearn.neighbors import BallTree

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# the model so far outputs 3D feature maps (height, width, features)


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# model.summary()

model.load_weights('five_layers.h5')

# get the image name list
ImageNameList = []
f = open("imagelist.txt", "r")
while True:  
    line = f.readline()
    if line:
        ImageNameList.append(line)
    else:
        break
f.close()

# get all the information from 5613 images
x_test = get_test_from_dir("image")
x_test = x_test.astype('float32')
x_test /= 255
print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'train samples')
pre=model.predict_proba(x_test)

# create ball tree
tree = BallTree(pre)

# make the test
TestImageData,TestImageName = get_testpic_from_dir("test")# ImageData is the array of image ;ImageName is the array of the names of the picture in the ImageData
TestImageData = TestImageData.astype('float32')
TestImageData /= 255
query_matrix = model.predict_proba(TestImageData)

#print(matrix)
num_query = len(TestImageName)
f=open('result.txt','w') 

for i in range(0, num_query):
    TargetPre,TargetSuf = TestImageName[i].split('_')
    TargetID, TargetFormat = TargetSuf.split('.')
    f.write(TargetID + ':')
    distance, inden = tree.query( [query_matrix[i]], k = 11)
    for j in range(1,11):
        HitString = ImageNameList[ inden[0][j] ]
        HitPre,HitSuf = HitString.split('_')
        HitID, HitFormat = HitSuf.split('.')
        f.write(HitID)
        if j == 10:
            f.write('\n')
        else:
            f.write(',')

f.close()
    

