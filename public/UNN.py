import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

seed = 42
np.random.seed = seed

# 样本图片大小
IMG_WIDTH = 584
IMG_HEIGHT = 584
IMG_CHANNELS = 3

# 数据集路径
DATA_PATH = '../../data/DRIVE/584_584/'

X_train = np.load(DATA_PATH + 'X_train.npy')
Y_train = np.load(DATA_PATH + 'Y_train.npy')
X_test = np.load(DATA_PATH + 'X_test.npy')
Y_test = np.load(DATA_PATH + 'Y_test.npy')

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)


# UNet输入模块
def InputBlock(input, filters, kernel_size=3, strides=1, padding='same'):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(input)  # 卷积块1
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                  activation='relu')(conv_1)  # 卷积块2


# 收缩路径模块
def ContractingPathBlock(input, filters, kernel_size=3, strides=1, padding='same'):
    down_sampling = tf.keras.layers.MaxPool2D((2, 2))(input)  # 最大池化
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                    activation='relu')(down_sampling)  # 卷积块1
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                  activation='relu')(conv_1)  # 卷积块2


# 扩张（恢复）路径模块
def ExpansivePathBlock(input, con_feature, filters, tran_filters, kernel_size=3, tran_kernel_size=2, strides=1,
                       tran_strides=2, padding='same', tran_padding='same'):
    upsampling = tf.keras.layers.Conv2DTranspose(filters=tran_filters, kernel_size=tran_kernel_size,
                                                 strides=tran_strides, padding=tran_padding)(input)  # 上采样（转置卷积方式）

    padding_h = (con_feature.shape)[1] - (upsampling.shape)[1]
    padding_w = (con_feature.shape)[2] - (upsampling.shape)[2]
    upsampling = img = tf.pad(upsampling, ((0, 0), (0, padding_h), (0, padding_w), (0, 0)), 'constant')
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

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(s)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    up6 = tf.pad(up6, ((0, 0), (0, 1), (0, 1), (0, 0)), 'constant')
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    return tf.keras.Model(inputs=[inputs], outputs=[conv10])


model = UNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
results = model.fit(X_train, Y_train, batch_size=1, epochs=10)

Y_pred = model.predict(X_test, batch_size=5, verbose=1)
Y_pred = np.array(Y_pred > 0, dtype="int").flatten()
Y_test = np.array(Y_test, dtype="int").flatten()
F1 = f1_score(Y_test, Y_pred, labels=None, average='binary', sample_weight=None)
print(">> F1-Score = {:.2f}%".format(np.mean(F1 * 100)))
