"""
detection.py
"""

import os
import struct
from pathlib import Path
from pprint import pprint
import cv2
from PIL import Image
import numpy as np
from keras import backend as k
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix

from model import MODEL
from tools.tools import resize2, focuse_image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# k.set_image_dim_ordering('th')
INPUT_PATH = Path('../../../data/input/License/Train')
OUTPUT_PATH = Path('../../../data/output/cs542/output')
if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)


def split_image(path):
    print(path)
    gridx, gridy = 30, 30
    img = Image.open(path)
    img = img.convert('RGB')
    img = focuse_image(img)     # focus img to center
    img = resize2(img)

    img_array = np.array(img)[np.newaxis, :]
    img_array = img_array.astype(np.float32)
    img_array /= 255
    print(img_array.shape)
    return img_array


def count_array(array):
    data = array[:, 1]
    score = data[0]
    print(score, score > 0.5)
    return score, score > 0.5, data


def best_model(model_path: Path):
    best_precision = 0
    best_path = 'latest_model.h5'
    for path in model_path.glob('*.h5'):
        if 'ckpt_model' in path.stem:
            # print(path.stem.split('-'))
            val_precision = float(path.stem.split('-')[1])
            if val_precision > best_precision:
                best_precision = val_precision
                best_path = path
    return str(best_path)


def predict():
    paths_list = [INPUT_PATH.glob('**/*.jpg')]
    model = MODEL(input_shape=(None, None, 3))
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model_path = best_model(Path('../../../data/output/cs542/models'))
    model.load_weights(model_path)
    # model.load_weights('../../../data/output/cs542/models/latest_model.h5')
    pprint(model.get_weights()[-1])

    good_output = OUTPUT_PATH / 'Good_Images'
    bad_output = OUTPUT_PATH / 'Bad_Images'
    re_output = OUTPUT_PATH / 'Re_Images'

    for p in [good_output, bad_output, re_output]:
        if not p.exists():
            p.mkdir(parents=True)

    for paths in paths_list:
        for path in paths:
            try:
                img_array, rangex, rangey = split_image(path)
            except (struct.error, OSError) as e:
                print('Error: %s, path: %s' % (e, path))
                continue
            score, flag, data = count_array(model.predict(img_array), rangex, rangey)
            img = Image.open(path)
            # img = img.convert('RGB')
            df = DataFrame(data)
            if path.parent.name == 'Good':
                img.save(str(good_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(good_output / ('{:.4f}.csv'.format(score))))
            elif path.parent.name == 'Bad':
                img.save(str(bad_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(bad_output / ('{:.4f}.csv'.format(score))))
            else:
                img.save(str(re_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(re_output / ('{:.4f}.csv'.format(score))))


def test():
    paths_list = [INPUT_PATH.glob('**/*.jpg')]
    model = MODEL(input_shape=(None, None, 3))
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model_path = best_model(Path('../../../data/output/total_image/models'))
    model.load_weights(model_path)
    # model.load_weights('../../../data/output/cs542/models/latest_model.h5')
    pprint(model.get_weights()[-1])

    y_true, y_pred = [], []
    for paths in paths_list:
        for path in paths:
            try:
                img_array = split_image(path)
            except (struct.error, OSError) as e:
                print('Error: %s, path: %s' % (e, path))
                continue
            if path.parent.name == 'Bad_License':
                y_true.append(1)
            elif path.parent.name == 'Good_License':
                y_true.append(0)
            else:
                # continue
                y_true.append(0)
            score, flag, data = count_array(model.predict(img_array))
            if score > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    # predict()
    test()
