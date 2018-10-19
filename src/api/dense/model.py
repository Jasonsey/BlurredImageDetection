from keras.models import Sequential
from keras.layers import Dense, Dropout

from config import Config


def linear_model(input_shape=(None, 3)):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.summary()
    return model

MODEL = linear_model
