# -*- coding:utf-8 -*- 
#!/usr/bin/env python3.6
# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""model train and test entry"""
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


# the gpu device input from command line or default configuration file  
os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES if not args.gpu else args.gpu


def main():
    """main entry of the project

    Arguments:
        command: input from comman line. Here are all possible values:
            train_all: train and test all of the 3 models
            train_cnn: only train and test the CNN model
            test_cnn: test the CNN model
            train_tree: train and test the decision tree model
            test_tree: test the decision tree model
            train_stacking: train and test the stacking model
            test_staacking: test the stacking model
            server: start the detection service
            client: start the detection client, and it will show the test results

    Returns:
        None
    """
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
