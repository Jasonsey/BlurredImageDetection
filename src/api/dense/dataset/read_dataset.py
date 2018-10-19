import sys 

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sys.path.append('..')
from tools.tools import get_imginfo
from config import Config


# TODO: 先不使用focuse功能，再对比带focuse功能的效果
def load_dataset(paths: list):
    data = []
    labels = []
    for i in range(len(paths)):
        path = Path(paths[i])
        for pa in path.glob('**/*.jpg'):
            labels.append(i)
            imginfo = list(get_imginfo(pa))
            data.append(imginfo)

    data = np.array(data)
    labels = np.array(labels)
    return shuffle(data, labels, random_state=2)


def split_dataset(data, labels):
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=2)
    return x_train, y_train, x_test, y_test
    

def read_dataset(paths: list):
    assert len(paths) == Config.num_class, 'length of paths should be %s, but get %s' % (NUM_CLASS, len(paths))

    data, labels = load_dataset(paths)
    x_train, y_train, x_test, y_test = split_dataset(data, labels)

    mean = x_train.mean(axis=0)
    x_train -= mean
    std = x_train.std(axis=0)
    x_train /= std

    x_test -= mean
    x_test /= std
    print('All Dataset Read!')

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    data, labels = load_dataset(['data/input/License/temp/Good_License', 'data/input/License/temp/Bad_License'])
    split_dataset(data, labels)