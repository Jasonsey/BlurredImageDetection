# -*- coding:utf-8 -*- 
#!/usr/bin/env python3.6
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
from utils.params import args
from deploy.thrift_client import test as client
from deploy.thrift_server import main as server


os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES if not args.gpu else args.gpu


def main():
    command = args.command
    work_flows = {
        'train_all': [cnn_train, cnn_test, tree_train, tree_test, stacking_train, stacking_test],
        'train_cnn': [cnn_train, cnn_test],
        'test_cnn': [cnn_test],
        'train_tree': [tree_train, tree_test],
        'test_tree': [tree_test],
        'train_stacking': [stacking_train, stacking_test],
        'test_stacking': [stacking_test],
        'server': [server],
        'client': [client]
    }

    if command not in work_flows:
        print('Please Check your command')
    else:
        for func in work_flows[command]:
            func()


if __name__ == '__main__':
    main()
