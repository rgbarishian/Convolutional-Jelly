import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import optimizers, backend as K
from keras.models import Sequential, save_model, load_model, model_from_json
from keras.layers import Activation, GlobalMaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam, rmsprop, Adadelta, adamax
from keras.metrics import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix
from time import time
from keras.utils import plot_model

#train_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Train_Valid_Mix'#mixed train path
train_path = '/home/ryan/Documents/DATA/CNN/Denoised/Train'
valid_path = '/home/ryan/Documents/DATA/CNN/Denoised/Valid'
test_path = '/home/ryan/Documents/DATA/Galaxy_Zoo/SmallBatch_TestBW'

train_batches = ImageDataGenerator(rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=360,
        width_shift_range=.25,
        height_shift_range=.25,
        fill_mode='wrap').flow_from_directory(train_path, target_size=(424,424), classes=['Not_Jellies','Jellies'], batch_size=35)
valid_batches = ImageDataGenerator(rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=360,
        width_shift_range=.15,
        height_shift_range=.15,
        fill_mode='wrap').flow_from_directory(valid_path, target_size=(424,424), classes=['Not_Jellies','Jellies'], batch_size=29)
test_batches = ImageDataGenerator(rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=360,
        width_shift_range=.15,
        height_shift_range=.15,
        fill_mode='wrap').flow_from_directory(test_path, target_size=(424,424), classes=['Not_Jellies','Jellies'], batch_size=22)

#LOAD PRE-TRAINED
"""
##Load in Pre-Trained weights
with open('model_in_json.json', 'r') as f:
    model_json = json.load(f)
model = model_from_json(model_json)
model.load_weights('simple_CNN.h5')
##
"""
"""
#code from https://arxiv.org/pdf/1709.02245.pdf
model = Sequential()
model.add(Conv2D(96,(8,8), input_shape=(424,424,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(.1))#my own addition
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dropout(.2))#my own addition
model.add(Conv2D(32,(4,4)))
model.add(Dense(2))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
"""

#from https://forums.fast.ai/t/my-dogs-vs-cats-models-always-have-0-5-accuracy-whats-wrong/1665
##Current Research Model##
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape=(424,424,3)))#3x3 is default
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# #model.add(Dropout(.1))#test
# model.add(Dense(32, activation='relu'))#test
# model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(.3))#test
# model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dropout(.3))
# model.add(Flatten(input_shape=(424,424,3)))
# model.add(BatchNormalization())
# model.add(Dense(2))
# model.add(Activation('softmax'))
################################
###EH 301 PROJECT###
# from keras.layers import Reshape
# model = Sequential()
# model.add(Conv2D(64,(3,3), input_shape=(424,424,3)))#3x3 is default
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# #model.add(Dropout(.1))#test
# model.add(Dense(32, activation='relu'))#test
# model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(.3))#test
# model.add(Conv2D(64,(3,3)))#input_shape=(424,424,3)
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3,3)))
# model.add(Dropout(.3))
# model.add(Flatten(input_shape=(424,424,3)))
# model.add(BatchNormalization())
# model.add(Dense(2))
# model.add(Activation('softmax'))

##condense
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(424,424,3), activation='relu'))#3x3 is default
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(.1))#test
model.add(Dense(32, activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))#input_shape=(424,424,3)
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.3))#test
model.add(Conv2D(64,(3,3), activation='relu'))#input_shape=(424,424,3)
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(.3))
model.add(Flatten(input_shape=(424,424,3)))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

#dense64,dense32,flatten,dense,softmax
##1585798155.4732356
#flatten,dense65,dense32,dense,softmax
##1585798704.7925284
#dense64,flatten,dense,softmax
##1585882810.7798576
#conv2d64,dense64,flatten,dense,softmax
##1585883555.6025758
#Default for research
##
################################
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)#saves plot

#Compile and run code
#model.summary()
myoptimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=myoptimizer, loss = 'mean_squared_error', metrics=['accuracy'] )
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1)
tensorboard= TensorBoard(log_dir="logs/{}".format(time()))
#tensorboard.add_graph(tf.summary.histogram)
model.fit_generator(train_batches, steps_per_epoch=5, validation_data=valid_batches, validation_steps=2, epochs=35, verbose=2, callbacks=[tensorboard])

"""
#Save Weights and model
model_json = model.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights('simple_CNN.h5')
"""

model.save_weights('Weights_CNN.h5')
model.save('Model_CNN.h5')
#Plot graph of accuracies
#plt.plot()

#test file
# filenames = train_batches.filenames
# nb_samples = len(filenames)
# predict = model.predict_generator(test_batches, steps=1 )
# print(predict)
# np.savetxt('test.csv', predict)