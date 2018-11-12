"""
detection.py
"""

import struct
from pathlib import Path
from pprint import pprint

import cv2
from PIL import Image
import numpy as np
from keras import backend as k
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

from utils.tools import resize2, focuse_image, get_imginfo
from dataset import read_dataset


def split_image(path):
    img = get_imginfo(path)

    img_array = np.array(img)[np.newaxis, :]
    img_array = img_array.astype(np.float32)
    img_array /= 255
    print(img_array.shape)
    return img_array


def count_array(array):
    print(array)
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
    print('Best Model: %s' % best_path.name)
    return str(best_path)


def predict(arrays):
    '''
    tree预测图片得分接口，输入arrays，返回对应(arrays.shape[0], 2)的得分
    '''
    model_path = '../data/output/decision_tree/models/train_model.pkl'
    model = joblib.load(model_path)
    results = model.predict_proba(arrays)
    return results


def test():
    input_path = Path('../data/input/License/Test')
    model_path = '../data/output/decision_tree/models/train_model.pkl'

    paths_list = [input_path / 'Good_License', input_path / 'Bad_License']
    model = joblib.load(model_path)

    data, labels = read_dataset.load_dataset(paths_list, random=False)
    y_ = model.predict_proba(data)
    y_ = y_[y_[:,0]>0]
    y_ = y_[y_[:,0]<1]
    pprint(y_)
    y_pred = model.predict(data)
    y_true = labels

    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    pass
