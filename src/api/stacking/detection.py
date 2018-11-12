"""
detection.py
"""

import struct
from pathlib import Path
from pprint import pprint
import asyncio

import cv2
from PIL import Image
import numpy as np
from keras import backend as k
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

from utils.tools import resize2, focuse_image, get_imginfo_by_array
from dataset.read_dataset import load_dataset3
from api.decision_tree.detection import predict as tree_predict
from api.total_image.detection import predict as cnn_predict


def load_img_score(arrays: list):
    async def get_info_array(array):
        info = get_imginfo_by_array(array)
        img = Image.fromarray(array)
        img = focuse_image(img)
        img = resize2(img)
        arr = np.asarray(img)
        return info, arr

    results = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for array in arrays:
        results.append(asyncio.ensure_future(get_info_array(array)))
    loop.run_until_complete(asyncio.wait(results))
    
    imginfo = []
    imgarray = []
    for result in results:
        info, arr = result.result()
        imginfo.append(info)
        imgarray.append(arr)
        
    imginfo = np.asarray(imginfo)
    imgarray = np.asarray(imgarray)

    tree_score = tree_predict(imginfo)[:, 1].reshape((-1, 1))
    cnn_score = cnn_predict(imgarray)
    data = np.concatenate((tree_score, cnn_score), axis=1)
    return data


def predict(arrays: list):
    '''
    stacking预测图片得分接口，输入arrays，返回对应(arrays.shape[0], 2)的得分
    '''
    model_path = '../data/output/stacking/models/train_model.pkl'
    model = joblib.load(model_path)

    data = load_img_score(arrays)

    results = model.predict_proba(data)
    return results


def test():
    input_path = Path('../data/input/License/Test')
    model_path = '../data/output/stacking/models/train_model.pkl'

    paths_list = [input_path / 'Good_License', input_path / 'Bad_License']
    model = joblib.load(model_path)

    data, labels = load_dataset3(paths_list, random=False)

    # # test scale
    # data *= np.array([0.5, 0.5], dtype='float32')
    # y_pred = (data.sum(axis=1)>0.5).astype('float32')

    # y_ = model.predict_proba(data)
    # y_ = y_[y_[:,0]>0]
    # y_ = y_[y_[:,0]<1]
    # pprint(y_)
    y_pred = model.predict(data)
    y_true = labels

    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    pass
