from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras.applications import VGG16, ResNet50, Xception
from keras import regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf

import config


def gen_model(input_shape=(3, 30, 30)):
    model = Sequential()
    model.add(Conv2D(96, (7, 7), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(config.NUM_CLASS, activation='softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
    return model


def gen_model2(input_shape=(None, None, 3)):
    print(input_shape)
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


def gen_model3(input_shape=(3, 30, 30)):
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    for layer in conv_base.layers:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.NUM_CLASS))
    model.add(Activation('softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
    return model


def gen_model4(input_shape=(None, None, 3)):
    print(input_shape)
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    for layer in conv_base.layers:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation('softmax'))

    # learning_rate = 0.1
    # adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

    model.summary()
    return model


def gen_model5(input_shape=(None, None, 3)):
    print(input_shape)
    conv_base = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    for layer in conv_base.layers:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dense(2))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

    model.summary()
    return model


def gen_model6(input_shape=(None, None, 3)):
    '''
    Muti GPU Model of Model2, for tiny model, unefficent
    '''
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
