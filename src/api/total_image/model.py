# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""CNN model file"""
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras.applications import VGG16, ResNet50, Xception
from keras import regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf

import config


def gen_model2(input_shape=(None, None, 3)):
    """single gpu model"""
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation('sigmoid'))

    model.summary()
    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model


def gen_model6(input_shape=(None, None, 3)):
    """multi gpu model. if there is only one gpu, it will be degraded to a single GPU model""""
    print(input_shape)
    if config.GPUS == 1:
        return gen_model2(input_shape)

    with tf.device('/gpu:0'):
        x0 = Input(shape=input_shape, dtype='float32')
        x = Conv2D(64, (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.01))(x0)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    with tf.device('/gpu:1'):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(1, (3, 3), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        y = Activation('sigmoid')(x)

    model = Model(inputs=x0, outputs=y)
    model.summary()

    # model = multi_gpu_model(model, gpus=config.GPUS)
    # model.summary()

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model


# aliase
MODEL = gen_model6


if __name__ == '__main__':
    gen_model2((256, 256, 3))
