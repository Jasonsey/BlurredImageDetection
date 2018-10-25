"""
train.py
"""

import os
from pathlib import Path
from multiprocessing import cpu_count
from keras import backend as k
from easydict import EasyDict as edict

from tools.callbacks import callbacks
from tools.tools import init_path
from dataset.read_dataset import load_dataset, prepare_train_data, datagen
from model import MODEL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# k.set_image_dim_ordering('th')


def train(model, x_train, x_test, y_train, y_test, model_direction, pretrain_model=None):
    callbacks_list = callbacks(model_direction)
    if pretrain_model:
        model.load_weights(pretrain_model)
    print(model.get_weights()[-1])
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=500,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list)
    return model


def train2(model, train_generator, train_steps, x_test, y_test, model_direction, pretrain_model, epoch_size, batch_size):
    callbacks_list = callbacks(model_direction, epoch_size, batch_size)
    if pretrain_model:
        model.load_weights(pretrain_model)
    print(model.get_weights()[-1])

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=1000,
        validation_data=(x_test, y_test),
        verbose=1,
        workers=cpu_count(),
        use_multiprocessing=True,
        callbacks=callbacks_list)
    return model


def test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    test_image = x_test[0:1]
    print(test_image.shape)

    print(model.predict(test_image))
    print(model.predict_classes(test_image))
    print(y_test[0:1])


def main():
    blur_directory = '../../../data/output/cs542/train/blurred/'
    clear_directory = '../../../data/output/cs542/train/clear/'
    model_direction = "../../../data/output/cs542/models/"
    pretrain_model = None
    init_path([model_direction])

    img_data_positive, labels_positive = load_dataset(Path(blur_directory), positive_label=True)
    img_data_negative, labels_negative = load_dataset(Path(clear_directory), positive_label=False,
                                                      length=len(labels_positive))
    dataset_dict = edict({
        'positive': {
            'img_data': img_data_positive,
            'labels': labels_positive
        },
        'negative': {
            'img_data': img_data_negative,
            'labels': labels_negative
        }
    })
    x_train, x_test, y_train, y_test, img_data = prepare_train_data(dataset_dict)

    model = MODEL(img_data[0].shape)
    model = train(model, x_train, x_test, y_train, y_test, model_direction, pretrain_model)
    test(model, x_test, y_test)


def main2():
    blur_directory = '../../../data/output/total_image/train/blurred/'
    clear_directory = '../../../data/output/total_image/train/clear/'
    model_direction = "../../../data/output/total_image/models/"
    batch_size = 64
    pretrain_model = None
    init_path([model_direction])


    img_data_positive, labels_positive = load_dataset(Path(blur_directory), positive_label=True)
    img_data_negative, labels_negative = load_dataset(Path(clear_directory), positive_label=False)
    dataset_dict = edict({
        'positive': {
            'img_data': img_data_positive,
            'labels': labels_positive
        },
        'negative': {
            'img_data': img_data_negative,
            'labels': labels_negative
        }
    })
    x_train, x_test, y_train, y_test, img_data = prepare_train_data(dataset_dict)

    model = MODEL(img_data[0].shape)

    train_generator, train_steps, epoch_size = datagen(x_train, y_train, batch_size)
    print('train_steps: %s' % train_steps)
    model = train2(model, train_generator, train_steps, x_test, y_test, model_direction, pretrain_model, epoch_size, batch_size)


if __name__ == '__main__':
    # main()
    main2()
