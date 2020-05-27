import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

seed = 42
np.random.seed = seed

# 样本图片大小
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# 数据集路径
DATA_PATH = '../data/bowl2018/'

# 数据加载
TRAIN_PATH = DATA_PATH + 'stage1_train/'  # 训练集路径
TEST_PATH = DATA_PATH + 'stage1_test/'  # 测试集路径

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# 构造训练集输入和输出（mask）
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  # Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    # mask
    for mask_file in next(os.walk(path + '/masks/'))[2]:  # os.walk()文件、目录遍历器,在目录树中游走输出在目录中的文件名
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)

    Y_train[n] = mask

# 构造测试集输入
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)


# 收缩路径模块
def ContractingPathBlock(input, filters, kernel_size=3, strides=1, padding='same'):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(input)  # 卷积块1
    conv_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(conv_1)  # 卷积块2
    return tf.keras.layers.MaxPool2D((2, 2))(conv_2)  # 最大池化


# UNet底部的处理模块
def BottomBlock(input, filters, kernel_size=3, strides=1, padding='same'):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(input)  # 卷积块1
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                  activation='relu')(conv_1)  # 卷积块2


# 扩张（恢复）路径模块
def ExpansivePathBlock(input, con_feature, filters, tran_filters, kernel_size=3, tran_kernel_size=2, strides=1,
                       tran_strides=2, padding='same', tran_padding='same'):
    upsampling = tf.keras.layers.Conv2DTranspose(filters=tran_filters, kernel_size=tran_kernel_size,
                                                 strides=tran_strides, padding=tran_padding)(input)  # 上采样（转置卷积方式）
    con_feature = tf.image.resize(con_feature, ((upsampling.shape)[1], (upsampling.shape)[2]),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 裁剪需要拼接的特征图
    concat_feature = tf.concat([con_feature, upsampling], axis=3)  # 拼接扩张层和收缩层的特征图（skip connection）
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(concat_feature)  # 卷积1
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                  activation='relu')(conv_1)  # 卷积2


# UNet网络架构
def UNet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    # contracting path
    con_1 = ContractingPathBlock(s, 64)
    con_2 = ContractingPathBlock(con_1, 128)
    con_3 = ContractingPathBlock(con_2, 256)
    con_4 = ContractingPathBlock(con_3, 512)

    # bottom block
    bott = BottomBlock(con_4, 1024)

    # expansive path
    exp_4 = ExpansivePathBlock(bott, con_4, 512, 512)
    exp_3 = ExpansivePathBlock(exp_4, con_3, 256, 256)
    exp_2 = ExpansivePathBlock(exp_3, con_2, 128, 128)
    exp_1 = ExpansivePathBlock(exp_2, con_1, 64, 64)

    outputs = tf.keras.layers.Conv2D(1, 1)(exp_1)  # 最终输出

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


model = UNet(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results = model.fit(X_train, Y_train, batch_size=10, epochs=10)
