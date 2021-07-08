import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import optimizers
from keras.optimizers import Adam, rmsprop, Adadelta, adamax
from keras.models import Model, Sequential, save_model, load_model, model_from_json
from keras.layers import Activation, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.metrics import categorical_crossentropy, mean_squared_error, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import plot_model
from time import time
# Location of Images
train_path = '/home/ryan/Documents/DATA/CNN/Plain/Train'
valid_path = '/home/ryan/Documents/DATA/CNN/Plain/Valid'
test_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Test'
#Import and Preprocessing
train_batches = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360,
    width_shift_range=.25,
    height_shift_range=.25,
    fill_mode='wrap').flow_from_directory(train_path, target_size=(424, 424), classes=['Not_Jellies', 'Jellies'], batch_size=35,shuffle=True,seed=58)
valid_batches = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360,
    width_shift_range=.25,
    height_shift_range=.25,
    fill_mode='wrap').flow_from_directory(valid_path, target_size=(424, 424), classes=['Not_Jellies', 'Jellies'], batch_size=29,shuffle=True,seed=7)
test_batches = ImageDataGenerator(
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=360,
    width_shift_range=.25,
    height_shift_range=.25,
    fill_mode='wrap').flow_from_directory(test_path, target_size=(424, 424), batch_size=15)

#Network
input_img = Input(shape=(424,424,3))
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='Conv1', trainable=True)(input_img)
x = MaxPooling2D((2, 2), padding='same', name = 'Pool1', trainable=True)(x)
x = Dense(64, activation='relu', name = 'Dense1', trainable=True)(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='Conv2', trainable=True)(x)
x = MaxPooling2D((2, 2), padding='same', name='Pool2', trainable=True)(x)
x = Dense(32, activation='relu', name='Dense2', trainable=True)(x)
x = Dropout(.3)(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='Conv3', trainable=True)(x)
x = MaxPooling2D((2, 2), padding='same', name='Pool3', trainable=True)(x)
x = Dropout(.3)(x)
x = Conv2D(64,(3,3), activation='relu', trainable=True)(x)
x = MaxPooling2D(pool_size=(3,3), trainable=True)(x)

x = Dropout(.3)(x)
x = Flatten(input_shape=(424,424,3))(x)
x = BatchNormalization()(x)
X = Dense(2, activation='softmax')(x)

model = Model(input_img, X)

#load autoencoding model
model.load_weights('/home/ryan/Documents/Unsupervised_Jelly/Autoenconding/Decoded.h5', by_name=True)

#load current model weights when unfreezing layer by layer
#model.load_weights('/home/ryan/Documents/Galaxy_Zoo/Good_Networks/Weights_CNN.h5')

#print(model.summary())
myoptimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=myoptimizer, loss = 'mean_squared_error', metrics=['accuracy'] )
tensorboard= TensorBoard(log_dir="AutoConv_logs/{}".format(time()))
model.fit_generator(train_batches, steps_per_epoch=3, validation_data=valid_batches, validation_steps=2, epochs=25, verbose=2, callbacks=[tensorboard])

model.save_weights('Weights_CNN.h5')
model.save('Model_CNN.h5')
