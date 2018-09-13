import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from pprint import pprint

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import backend as k
k.set_image_dim_ordering('th')

input_path = Path('../data/input/source')
input_good = Path('../data/input/Good')
input_bad = Path('../data/input/Bad')
liscense_path = Path('../data/input/liscense')


size = 15
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size


def generate_test_image():
    paths = input_path.glob('**/*.jpg')
    i = 0
    for path in paths:
        img = cv2.imread(str(path))
        cv2.imwrite(str(input_good / ('%s.jpg' % i)), img)
        img = cv2.filter2D(img, -1, kernel_motion_blur)
        cv2.imwrite(str(input_bad / ('%s.jpg' % i)), img)
        i += 1


def resize_image(img):
    img = img.resize((90, 90), Image.ANTIALIAS)
    return img


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width / w, height / h) if width < height else (width / h, height / w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def split_image(path):
    gridx, gridy = 30, 30
    path = str(path)
    img = Image.open(path)
    img = focuse_image(img)     # focus img to center
    rangex, rangey = img.width / gridx, img.height / gridy

    img_data_list = []
    for x in range(rangex):
        for y in range(rangey):
            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = np.asarray(img.crop(bbox))
            slice_bit = cv2.cvtColor(slice_bit, cv2.COLOR_RGB2BGR)
            slice_bit = np.swapaxes(slice_bit, 0, 2)
            img_data_list.append(slice_bit)
    img_array = np.array(img_data_list)
    img_array = img_array.astype(np.float32)
    img_array /= 255
    print(img_array.shape)
    return img_array


def gen_model():
    input_shape = (3, 30, 30)
    model = Sequential()
    model.add(Convolution2D(96, 7,7,input_shape=input_shape))
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
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def count_array(array):
    positive = array.sum()
    negative = len(array) - positive
    print(positive, negative, positive > negative)


def predict():
    # paths_list = [input_good.glob('**/*.jpg'), input_bad.glob('**/*.jpg')]
    paths_list = [liscense_path.glob('**/*.jpg')]
    model = gen_model()
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model.load_weights('motionblur.h5')
    pprint(model.get_weights()[-1])

    for paths in paths_list:
        for path in paths:
            img_array = split_image(path)
            count_array(model.predict_classes(img_array))
            img = Image.open(path)
            img.show()
            input('OK?:')



if __name__ == '__main__':
    # generate_test_image()
    predict()
