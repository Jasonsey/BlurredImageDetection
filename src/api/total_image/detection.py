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
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import tensorflow as tf
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix

from utils.tools import resize2, focuse_image, init_path
from dataset import read_dataset


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


def best_model(model_path: Path, standard='val_f1'):
    best_score = 0
    best_path = 'latest_model.h5'
    for path in model_path.glob('*.h5'):
        if 'ckpt_model' in path.stem and standard in path.stem:
            # print(path.stem.split('-'))
            score = float(path.stem.split('-')[1])
            if score > best_score:
                best_score = score
                best_path = path
    print('Best Model: %s' % best_path.name)
    return str(best_path)


def predict(arrays):
    '''
    cnn预测图片得分接口，输入arrays，返回对应(arrays.shape[0], 2)的得分
    '''
    # config TF Session
    config = tf.ConfigProto()  
    # config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    set_session(session)

    model_path = best_model(Path('../data/output/total_image/models'))
    model = load_model(model_path)

    results = model.predict(arrays/255)
    k.clear_session()
    return results


def test():
    input_path = Path('../data/input/License/Test')
    model_path = best_model(Path('../data/output/total_image/models'))

    paths_list = [input_path / 'Good_License', input_path / 'Bad_License']
    model = load_model(model_path)
    pprint(model.get_weights()[-1])

    data, labels = read_dataset.load_dataset2(paths_list, random=False)
    y_pred = model.predict_classes(data)
    y_true = labels

    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    # predict()
    # test()
    pass
