# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""the dicison tree model's detection api"""
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


def predict(arrays):
    """the dicison tree model's detection api
    
    Arguments:
        arrays: 2D np.ndarray with shape (None, 3)
    
    Returns:
        1D np.ndarray with shape (None, 1)
    """
    model_path = '../data/output/decision_tree/models/train_model.pkl'
    model = joblib.load(model_path)
    results = model.predict_proba(arrays)
    return results


def test():
    """pipline for detection the dataset with decision tree model"""
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
