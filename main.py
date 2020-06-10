from model import unet
from data.preprossing import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import *
import numpy as np

# 样本图片大小
IMG_WIDTH = 584
IMG_HEIGHT = 584
IMG_CHANNELS = 3

X_train = np.load('../data/DRIVE/X_train.npy')
Y_train = np.load('../data/DRIVE/Y_train.npy')

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, '', 'images', '1st_manual', data_gen_args, save_to_dir=None)

model = unet.UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(myGene, steps_per_epoch=300, epochs=1)

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)
