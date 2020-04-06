import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.models import Sequential, save_model, load_model, model_from_json
from keras.layers import Activation, GlobalMaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout, MaxoutDense
from keras.optimizers import Adam, rmsprop, Adadelta, adamax
from keras.metrics import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.callbacks import EarlyStopping, TensorBoard
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import json
from time import time

train_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Train_Independent'
valid_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Valid'
test_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Test'
#Pull images from directories
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(424,424), classes=['Not_Jellies','Poss_Jellies'], batch_size=16)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(424,424), classes=['Not_Jellies','Poss_Jellies'], batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(424,424), batch_size=15)

#NN Model
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(424,424,3)))#3x3 is default
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(.1))#test
model.add(Dense(32, activation='relu'))#test
model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.2))#test
model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(.2))
model.add(Flatten(input_shape=(424,424,3)))
model.add(MaxoutDense(128))###testing
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
tensorboard= TensorBoard(log_dir="logs/{}".format(time())) #allows me to visualize results
model.fit_generator(train_batches, steps_per_epoch=2, validation_data=valid_batches, validation_steps=2, epochs=40, verbose=2, callbacks=[tensorboard])

model.save('simple_CNN.h5')