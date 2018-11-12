import json
import io
import time
import asyncio
from pathlib import Path
from pprint import pprint

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from multiprocessing import Pool

from api.thrift_api.interface import blur_detection  # 引入客户端类
import config


class TransportClient(object):
    """
    传输一个文件到服务端并返回服务端返回的数据
    """
    def __init__(self, host='0.0.0.0', port=9099):
        self.host = host
        self.port = port
        self.client = None
        self.transport = None
        self.open()

    def send(self, imgbyte):
        try:
            res = self.client.reco(imgbyte)
            res = json.loads(res)
        except Thrift.TException as e:
            res = {'status': 0, 'msg': e, 'data': ''}
        return res

    def open(self):
        transport = TSocket.TSocket(self.host, self.port)
        # 选择传输层，这块要和服务端的设置一致
        self.transport = TTransport.TBufferedTransport(transport)
        # 选择传输协议，这个也要和服务端保持一致，否则无法通信
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        # 创建客户端
        self.client = blur_detection.Client(protocol)
        self.transport.open()

    def close(self):
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.transport:
            self.close()
        return False    # 如果出现异常，则异常传递到外围


def load_dataset(paths: list):
    '''
    return data: list, labels: array
    '''
    assert len(paths) == 2, 'len of args should be 2'

    data, labels = [], []
    for i in range(len(paths)):
        path = Path(paths[i])
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for pa in path.glob('**/*.jpg'):
            print(pa)
            results.append(asyncio.ensure_future(get_imgarray(pa)))
            labels.append(i)
        loop.run_until_complete(asyncio.wait(results))
        for result in results:
            data.append(result.result())

    labels = np.array(labels, dtype='float32')
    print('Blur: %s, Clear: %s' % ((labels==1).sum(), (labels==0).sum()))
    return data, labels


async def get_imgarray(path: Path):
    with open(path, mode='rb') as f:
        res = f.read()
    return res


def get_result(data: bytes):
    with TransportClient(host=config.THRIFT_HOST, port=config.THRIFT_PORT) as client:
        res = client.send(data)
    return res


def test():
    input_path = Path('../data/input/License/Test')
    paths_list = [input_path / 'Good_License', input_path / 'Bad_License']

    data, labels = load_dataset(paths_list)
    y_pred, y_true = [], []

    start = time.time()
    pool = Pool()
    results = pool.map(get_result, data)
    
    for i in range(len(labels)):
        res = results[i]
        if res['status'] == 1:
            y_pred.append(res['data']['pred'])
            y_true.append(labels[i])
        else:
            print(res['msg'])
    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))
    print(time.time() - start)


if __name__ == '__main__':
    test()
    