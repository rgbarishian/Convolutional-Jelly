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

##Network##
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

print(model.summary())

#Compile and run code
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

#test against test directory
# filenames = train_batches.filenames
# nb_samples = len(filenames)
# predict = model.predict_generator(test_batches, steps=1 )
# print(predict)
# np.savetxt('test.csv', predict)
