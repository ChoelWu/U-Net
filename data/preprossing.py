import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import *
import numpy as np


def trainGenerator(x, y, aug_dict):
    image_data_generator = ImageDataGenerator(**aug_dict)
    image_generator = image_data_generator.flow(x, y, 2)
    for (img, label) in image_generator:
        yield (img, label)
