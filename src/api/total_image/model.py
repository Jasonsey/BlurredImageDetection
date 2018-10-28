from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras.applications import VGG16, ResNet50, Xception
from keras import regularizers


num_classes = 2


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
    model.add(Dense(num_classes, activation='softmax'))

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
    model.summary()
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Activation('softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
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
    model.add(Dense(num_classes))
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
    model.add(Dense(2))

    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

    model.summary()
    return model



# aliase
MODEL = gen_model5

if __name__ == '__main__':
    gen_model5((256, 256, 3))
