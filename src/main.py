#!/usr/bin/env python
"""
模型训练、测试文件
"""


import os
from pathlib import Path

from api.decision_tree.train import main as tree_train
from api.decision_tree.detection import predict as tree_predict
from api.decision_tree.detection import test as tree_test

from api.total_image.train import main as cnn_train
from api.total_image.detection import predict as cnn_predict
from api.total_image.detection import test as cnn_test

from api.stacking.train import main as stacking_train
from api.stacking.detection import predict as stacking_predict
from api.stacking.detection import test as stacking_test

import config


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
flag = 5

if __name__ == '__main__':
    if flag == 0:
        tree_train()
    elif flag == 1:
        tree_test()
    elif flag == 2:
        cnn_train()
    elif flag == 3:
        cnn_test()
    elif flag == 4:
        stacking_train()
    elif flag == 5:
        stacking_test()
