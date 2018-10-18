import os
from pathlib import Path
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from src.api.dense.tools.tools import get_imginfo


NUM_CLASS = 2


# TODO: 先不使用focuse功能，再对比带focuse功能的效果
def load_dataset(paths: list):
    assert len(paths) == NUM_CLASS, 'length of paths should be %s, but get %s' % (NUM_CLASS, len(paths))

    data = []
    labels = []
    for i in range(len(paths)):
        path = Path(paths[i])
        for pa in path.glob('*.jpg'):
            labels.append(i)
            imginfo = list(get_imginfo(pa))
            data.append(imginfo)

    data = np.array(data)
    labels = np.array(labels)
    return data,labels 


def split_dataset(data, labels):
    labels = np_utils.to_categorical(labels, NUM_CLASS)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=2)
    


if __name__ == '__main__':
    data, labels = load_dataset(['data/input/License/temp/Good_License', 'data/input/License/temp/Bad_License'])
    split_dataset(data, labels)