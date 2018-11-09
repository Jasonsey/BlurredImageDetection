import os
import numpy as np

from api.cv2_api.detection import predict as cv2_predict
from api.stacking.detection import predict as net_predict
import config


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES


def predict(arrays: list):
    '''
    针对服务提供的接口
    '''
    cv2_scores = cv2_predict(arrays)

    net_arrays = [arrays[i] for i in range(len(cv2_scores)) if cv2_scores[i] < 1400]    # 经验值，高清图片阈值
    net_scores = net_predict(net_arrays)[:, 1].tolist()

    scores = []
    for i in range(len(cv2_scores)):
        if cv2_scores[i] < 1400:
            try:
                score = net_scores.pop(0)
            except IndexError as e:     # 理论上不会异常，以防万一
                print(e)
                score = 0.0
        else:
            score = 0.0
        scores.append(score)
    return np.asarray(scores, dtype='float32')
            