import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import *
import numpy as np


def adjustData(img, label, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
        new_label = np.zeros(label.shape + (num_class,))
        for i in range(num_class):
            new_label[label == i, i] = 1
        new_label = np.reshape(new_label, (new_label.shape[0], new_label.shape[1] * new_label.shape[2],
                                         new_label.shape[3])) if flag_multi_class else np.reshape(new_label, (
        new_label.shape[0] * new_label.shape[1], new_label.shape[2]))
        label = new_label
    elif (np.max(img) > 1):
        img = img / 255
        label = label / 255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (img, label)


def trainGenerator(batch_size, train_path, image_folder, label_folder, aug_dict, image_color_mode="grayscale",
                   label_color_mode="grayscale", image_save_prefix="image", label_save_prefix="label",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        train_path,
        # classes=[image_folder],
        # class_mode=None,
        # color_mode=image_color_mode,
        # target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    label_generator = label_datagen.flow(
        train_path,
        # classes=[label_folder],
        # class_mode=None,
        # color_mode=label_color_mode,
        # target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, label_generator)
    for (img, label) in train_generator:
        img, label = adjustData(img, label, flag_multi_class, num_class)
        yield (img, label)

