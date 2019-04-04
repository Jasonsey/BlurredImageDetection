# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""train the decision tree model"""
from pathlib import Path
from easydict import EasyDict
from sklearn import tree, metrics
from sklearn.externals import joblib

from utils.tools import init_path
from dataset.read_dataset import read_dataset


def train(model, input_dataset: EasyDict, output_path):
    """train the decision tree model

    Arguments:
        model: the untrained decision tree model
        input_dataset: a EasyDict that consists of train data set and train labels
        output_path: where the trained decision tree model will be saved
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
    """pipline of reading data set and trianing decision model"""
    blur_path = '../data/input/License/Train/Bad_License/'
    clear_path = '../data/input/License/Train/Good_License/'
    output_path = '../data/output/decision_tree/models'
    pretrain = None

    init_path([output_path])

    input_dataset = read_dataset([clear_path, blur_path])
    model = tree.DecisionTreeClassifier(max_depth=3)
    model = train(model, input_dataset, output_path)


if __name__ == '__main__':
    main()
