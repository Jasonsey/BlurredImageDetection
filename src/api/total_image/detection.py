# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""the CNN model's detection file"""
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
import config as project_config


def best_model(model_path: Path, standard='val_f1'):
    """find the best model from a given path

    Arguments:
        model_path: pathlib.Path where the cache model saved
        standard: criteria for finding the best model

    Returns:
        a string of the best model's path
    """
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
    """the CNN model's prediction api
    
    Arguments:
        arrays: 4D np.ndarray of images
    
    Returns:
        2D np.ndarray with shape (None, 2)
    """
    # config TF Session
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True    # 解决多进程下GPU cuda访问异常问题
    config.gpu_options.per_process_gpu_memory_fraction = project_config.PREDICT_GPU_MEMORY
    session = tf.Session(config=config)
    set_session(session)

    model_path = best_model(Path('../data/output/total_image/models'))
    model = load_model(model_path)

    results = model.predict(arrays/255, batch_size=2)
    k.clear_session()
    return results


def test():
    """pipline for test the CNN model"""
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
    pass
