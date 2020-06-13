from model import unet
from excu import train
from data.preprossing import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import *
import numpy as np


X_train = np.load('../data/DRIVE/584_584/X_train.npy')
Y_train = np.load('../data/DRIVE/584_584/Y_train.npy')
Y_test = np.load('../data/DRIVE/Y_test.npy')

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(1, '../data/DRIVE/training', 'images', '1st_manual', data_gen_args, save_to_dir=None)

model = unet.UNet(input_shape=(584, 565, 3))
model.summary()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("../data/DRIVE/test/")
results = model.predict_generator(testGene, 20, verbose=1)
print(results[0])
np.save('results.npy', results)

saveResult("../data/DRIVE/test/", results)

Y_test = np.array(Y_test, dtype="int").flatten()
results = np.array(results > 0, dtype="int").flatten()
F1 = f1_score(Y_test, results, labels=None, average='binary', sample_weight=None)
print(">> F1-Score = {:.2f}%".format(np.mean(F1 * 100)))
