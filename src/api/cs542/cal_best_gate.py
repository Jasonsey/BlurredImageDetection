from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from itertools import combinations


def cal_label(x_y):
    data_path = Path('../../../data/output/cs542/output')

    xx, yy = x_y
    print('Testing {}, {}'.format(xx, yy))
    x = xx / 100.0
    y = yy / 100.0
    assert y >= x, 'Y should be greater than X !'

    y_true, y_pred = [], []
    for path in data_path.glob('**/*.csv'):
        if path.parent.name == 'Bad_Images':
            y_true.append(1)
        else:
            y_true.append(0)

        df = pd.read_csv(path)
        p = (df > y).sum().sum()
        n = (df < x).sum().sum()
        if p >= n:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_true, y_pred, xx, yy


def main():
    gate_max = 100
    f1_array = np.zeros((gate_max, gate_max), dtype=np.float32)
    precision_array = np.zeros((gate_max, gate_max), dtype=np.float32)

    x_ys = combinations(range(gate_max), 2)
    pool = ThreadPool(cpu_count() * 2 + 2)
    results = pool.map(cal_label, x_ys)
    pool.close()
    pool.join()

    for result in results:
        y_true, y_pred, x, y = result
        f1_array[x, y] = f1_score(y_true, y_pred)
        precision_array[x, y] = precision_score(y_true, y_pred)

    re = np.where(f1_array == np.max(f1_array))
    for r in re:
        x, y = r
        f1 = f1_array[r]
        precision = precision_array[r]
        print('x: {0}, y: {1}, f1: {2}, precision: {3}'.format(x, y, f1, precision))


if __name__ == '__main__':
    main()

