from pathlib import Path
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint
from easydict import EasyDict as edict
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


num_classes = 2


def load_dataset(path: Path, positive_label: bool=True, length: int=0):
    img_data_list = []
    list_append = img_data_list.append
    i = 0
    for img_path in path.glob('*.jpg'):
        i += 1
        # todo resize
        input_img = cv2.imread(str(img_path))
        # input_img = cv2.resize(input_img, (60, 60), interpolation=cv2.INTER_CUBIC)
        input_img = np.swapaxes(input_img, 0, 2)
        list_append(input_img)
    if 0 < length < i:
        img_data_list = shuffle(img_data_list, random_state=2)
        img_data_list = img_data_list[:length]

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


def datagen(x_train, y_train, batch_size=128):
    train_steps = int(len(y_train) / batch_size) + 1
    
    train_datagen = ImageDataGenerator(
        rescale=None,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)
    return train_generator, train_steps