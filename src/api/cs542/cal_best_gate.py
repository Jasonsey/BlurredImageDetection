from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from multiprocessing import cpu_count
from itertools import combinations
from time import time
import asyncio


def cal_label(params):
    start = time()
    xx, yy, data = params
    x = xx / 100.0
    y = yy / 100.0
    assert y >= x, 'Y should be greater than X !'

    y_true, y_pred = [], []
    for k in data:
        loop = asyncio.get_event_loop()
        results = []
        results_append = results.append
        for df in data[k]:
            results_append(asyncio.ensure_future(cal_pred(x, y, df)))
        loop.run_until_complete(asyncio.wait(results))
        # loop.close()

        for result in results:
            if k == 'Bad_Images':
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(result.result())
    print('Ready for {}, {}: {}'.format(xx, yy, time()-start))
    # print(y_true, y_pred)
    return y_true, y_pred, xx, yy


async def cal_pred(x, y, df):
    p = (df > y).sum(axis=0).sum(axis=0)
    n = (df < x).sum(axis=0).sum(axis=0)
    if p >= n:
        return 1
    else:
        return 0


def read_csv(data_path: Path):
    data = {}
    for path in data_path.glob('**/*.csv'):
        df = pd.read_csv(path)
        if path.parent.name in data:
            data[path.parent.name].append(df)
        else:
            data[path.parent.name] = [df]
    return data


def main():
    data_path = Path('../../../data/output/cs542/output')
    gate_max = 100
    data = read_csv(data_path)

    f1_array = np.zeros((gate_max, gate_max), dtype=np.float32)
    precision_array = np.zeros((gate_max, gate_max), dtype=np.float32)

    params = []
    for x_y in combinations(range(gate_max), 2):
        xy = list(x_y)
        xy.append(data)
        params.append(tuple(xy))
    params = tuple(params)
    print('Ready for params !')

    pool = Pool(cpu_count() * 2 + 2)
    results = pool.map(cal_label, params)
    pool.close()
    pool.join()
    # print(list(results))
    for result in results:
        print(result)
        # input('OK?')
        y_true, y_pred, x, y = result
        f1_array[x, y] = f1_score(y_true, y_pred)
        precision_array[x, y] = precision_score(y_true, y_pred)
    np.save('precision.npy', precision_array)
    np.save('f1.npy', f1_array)

    re = np.where(f1_array == np.max(f1_array))
    print(re)
    f1 = f1_array[re]
    precision = precision_array[re]
    print('x: {0}, y: {1}, f1: {2}, precision: {3}'.format(re[0], re[1], f1, precision))


if __name__ == '__main__':
    main()

