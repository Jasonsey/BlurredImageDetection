import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from pprint import pprint
from keras import backend as k

from train import gen_model
from dataset2 import resize


k.set_image_dim_ordering('th')
input_path = Path('../../../data/input/License/Test')
output_path = Path('../../../data/output/cs542/output')
if not output_path.exists():
    output_path.mkdir(parents=True)


def resize_image(img):
    img = img.resize((90, 90), Image.ANTIALIAS)
    return img


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def split_image(path):
    gridx, gridy = 30, 30
    path = str(path)
    img = Image.open(path)
    img = resize(img)
    img = focuse_image(img)     # focus img to center
    rangex, rangey = img.width // gridx, img.height // gridy

    img_data_list = []
    list_append = img_data_list.append
    for x in range(rangex):
        for y in range(rangey):
            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = np.asarray(img.crop(bbox))
            slice_bit = cv2.cvtColor(slice_bit, cv2.COLOR_RGB2BGR)
            slice_bit = np.swapaxes(slice_bit, 0, 2)
            list_append(slice_bit)
    img_array = np.array(img_data_list)
    img_array = img_array.astype(np.float32)
    img_array /= 255
    print(img_array.shape)
    return img_array


def count_array(array):
    # pprint(array)
    mean = np.mean(array, axis=0)
    score = mean[1]
    print(score, score > 0.5)
    # positive = array.sum()
    # negative = len(array) - positive
    # score = float(positive) / (negative + positive)
    # print(positive, negative, positive > negative)
    # return score, positive > negative
    return score, score > 0.5


def predict():
    paths_list = [input_path.glob('**/*.jpg')]
    model = gen_model(input_shape=(3, 30, 30))
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model.load_weights('../../../data/output/cs542/models/ckpt_model.99-0.91.h5')
    # model.load_weights('../../../data/output/cs542/models/latest_model.h5')
    pprint(model.get_weights()[-1])

    good_output = output_path / 'Good_Images'
    bad_output = output_path / 'Bad_Images'

    for p in [good_output, bad_output]:
        if not p.exists():
            p.mkdir(parents=True)

    for paths in paths_list:
        for path in paths:
            img_array = split_image(path)
            score, flag = count_array(model.predict(img_array))
            img = Image.open(path)
            # img.show()
            # input('OK?:')
            img = img.convert('RGB')
            if flag:
                img.save(str(bad_output / ('{:.4f}.jpg'.format(score))))
            else:
                img.save(str(good_output / ('{:.3f}.jpg'.format(score))))


if __name__ == '__main__':
    predict()
