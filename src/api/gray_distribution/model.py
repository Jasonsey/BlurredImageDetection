from keras.models import Sequential
from keras.layers import Dense, Dropout

from hyperas.distributions import choice, uniform

from config import Config


def linear_model(input_shape=(None, 3)):
    model = Sequential()
    model.add(Dense({{choice([16, 32, 64, 128])}}, activation='relu', input_shape=input_shape))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([16, 32, 64, 128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([8, 16, 32, 64, 128])}}, activation='relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.summary()
    return model

MODEL = linear_model
