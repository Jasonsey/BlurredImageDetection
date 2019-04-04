# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""taining the stacking model"""
from pathlib import Path
from easydict import EasyDict
from sklearn import tree, metrics
from sklearn.externals import joblib

from utils.tools import init_path
from dataset.read_dataset import read_dataset3


def train(model, input_dataset: EasyDict, output_path):
    """train the stacking model
    
    Arguments:
        model: a untrained model
        input_dataset: an EasyDict which consists of train data an train labels
        output_path: where the trained model will be saved
    """
    model.fit(
        input_dataset.train.data,
        input_dataset.train.labels)
    y_pred = model.predict(input_dataset.test.data)
    print(metrics.classification_report(
        input_dataset.test.labels,
        y_pred,
        target_names=['清晰', '模糊']))

    output_path = Path(output_path) / 'train_model.pkl'
    joblib.dump(model, output_path)
    return model


def main():
    """pipline for training the stacking model"""
    blur_path = '../data/input/License/Train/Bad_License/'
    clear_path = '../data/input/License/Train/Good_License/'
    output_path = '../data/output/stacking/models'
    pretrain = None

    init_path([output_path])

    input_dataset = read_dataset3([clear_path, blur_path], use_cache=False)
    model = tree.DecisionTreeClassifier(max_depth=3)
    model = train(model, input_dataset, output_path)


if __name__ == '__main__':
    main()
