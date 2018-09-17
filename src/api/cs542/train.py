import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import callbacks
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.optimizers import Adam
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras import backend as k
from pathlib import Path
from pprint import pprint
from easydict import EasyDict as edict

from dataset import init_path


k.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)
num_classes = 2


def load_dataset(path: Path, positive_label: bool=True):
    img_data_list = []
    for img_path in path.glob('*.jpg'):
        input_img = cv2.imread(str(img_path))
        input_img = np.swapaxes(input_img, 0, 2)
        img_data_list.append(input_img)
    img_data = np.array(img_data_list, dtype=np.float32)
    img_data /= 255

    num_of_samples = img_data.shape[0]
    if positive_label:
        labels = np.ones((num_of_samples, ), dtype=np.int64)
        log = {'length of positive': len(labels)}
    else:
        labels = np.zeros((num_of_samples, ), dtype=np.int64)
        log = {'length of negative': len(labels)}
    pprint(log)
    return img_data, labels


def prepare_train_data(dataset_dict: edict):
    labels = np.concatenate((dataset_dict.positive.labels, dataset_dict.negative.labels), axis=0)
    img_data = np.concatenate((dataset_dict.positive.img_data, dataset_dict.negative.img_data), axis=0)
    labels = np_utils.to_categorical(labels, num_classes)
    x, y = shuffle(img_data, labels, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    return x_train, x_test, y_train, y_test, img_data


def gen_model(input_shape):
    # input_shape = img_data[0].shape
    # print(input_shape, 'input_shape')

    model = Sequential()
    model.add(Convolution2D(96, 7, 7, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # learning_rate = 0.001
    # adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    model.summary()
    # model.get_config()
    # model.layers[0].get_config()
    # model.layers[0].input_shape
    # model.layers[0].output_shape
    # model.layers[0].get_weights()
    # np.shape(model.layers[0].get_weights()[0])
    # model.layers[0].trainable

    return model


def train(model, x_train, x_test, y_train, y_test, model_direction):
    log_file = str(Path(model_direction) / 'model_train_new.csv')

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')
    csv_log = callbacks.CSVLogger(log_file)
    checkpoint = callbacks.ModelCheckpoint(model_direction, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                                 write_grads=False, write_images=False, embeddings_freq=0,
                                                 embeddings_layer_names=None, embeddings_metadata=None)
    callbacks_list = [csv_log, early_stopping, checkpoint, tensorboard_callback]
    model.fit(x_train, y_train, batch_size=128, nb_epoch=100, verbose=1, validation_data=(x_test, y_test),
              callbacks=callbacks_list)
    model.save(model_direction + "motionblur.h5")
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
    blur_directory = '../../../data/output/cs542/train/blur/'
    clear_directory = '../../../data/output/cs542/train/clear/'
    model_direction = "../../../data/output/cs542/models/"
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
            'label': labels_negative
        }
    })
    x_train, x_test, y_train, y_test, img_data = prepare_train_data(dataset_dict)
    model = gen_model(img_data[0].shape)
    model = train(model, x_train, x_test, y_train, y_test, model_direction)
    test(model, x_test, y_test)


if __name__ == '__main__':
    main()
