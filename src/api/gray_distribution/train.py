from pathlib import Path
from easydict import EasyDict
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from hyperas import optim

from tools.tools import init_path
from dataset.read_dataset import read_dataset
from tools.callbacks import callbacks


def model(input_dataset: EasyDict, output_path, pretrain=None):
    input_shape=input_dataset.train.data[0].shape

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

    callbacks_list = callbacks(output_path)
    if pretrain is not None:
        model.load_weights(pretrain)

    history = model.fit(
        input_dataset.train.data,
        input_dataset.train.labels,
        batch_size=128,
        epochs=500,
        verbose=1,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        callbacks=callbacks_list)
    print(model.get_weights()[0])
    best_acc = np.max(history.history['val_acc'])
    print('Best Val ACC: %s' % best_acc)
    return {'loss': -best_acc, 'status': STATUS_OK, 'model': model}


def data():
    blur_path = '../../../data/input/License/Train2/Bad_License/'
    clear_path = '../../../data/input/License/Train2/Good_License/'
    output_path = '../../../data/output/gray_distribution/'
    pretrain = None

    x_train, y_train, x_test, y_test = read_dataset([clear_path, blur_path])
    input_dataset = EasyDict({
        'train': {
            'data': x_train,
            'labels': y_train},
        'test': {
            'data': x_test,
            'labels': y_test}})
    return input_dataset, output_path, pretrain


def main():
    best_run, best_model, space = optim.minimize(
        model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=500,
        trials=Trials(),
        return_space=True)
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(space)


if __name__ == '__main__':
    main()
