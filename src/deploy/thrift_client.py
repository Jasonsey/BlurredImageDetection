# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""thrift clinent"""
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
    """transform an image to the thrift server and return the score.

    Arguments:
        host: a string of ip where the thrift server host
        port: an integer of the thrift server's port

    Examples:
    ```python
    with TransportClient(host=THRIFT_HOST, port=THRIFT_PORT) as client:
        res = client.send(one_image_array)   
    ``` 
    """
    def __init__(self, host='0.0.0.0', port=9099):
        self.host = host
        self.port = port
        self.client = None
        self.transport = None
        self.open()

    def send(self, imgbyte):
        """send an image with byte type
        
        Arguments:
            imgbyte: a byte type image

        Examples:
        ```python
        with open(path, 'rb') as f:
            imgbyte = f.read()
        ```
        """
        try:
            res = self.client.reco(imgbyte)
            res = json.loads(res)
        except Thrift.TException as e:
            res = {'status': 0, 'msg': e, 'data': ''}
        return res

    def open(self):
        """create the client and make it ready for transformation"""
        transport = TSocket.TSocket(self.host, self.port)
        # 选择传输层，这块要和服务端的设置一致
        self.transport = TTransport.TBufferedTransport(transport)
        # 选择传输协议，这个也要和服务端保持一致，否则无法通信
        protocol = TBinaryProtocol.TBinaryProtocol(self.transport)
        # 创建客户端
        self.client = blur_detection.Client(protocol)
        self.transport.open()

    def close(self):
        """close the client"""
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """the client will be closed if an exception occurs"""
        if self.transport:
            self.close()
        return False    # 如果出现异常，则异常传递到外围


def load_dataset(paths: list):
    """load images and its labels from disk.

    Arguments:
        paths: a list of string or pathlib.Path

    Returns:
        data: 4D np.ndarray
        labels: 1D np.ndarray
    '''"""
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
    """return an array of image"""
    with open(path, 'rb') as f:
    # with path.open(mode='rb') as f:
        res = f.read()
    return res


def get_result(data: bytes):
    """start the thrift client and transform data to server through the client"""
    with TransportClient(host=config.THRIFT_HOST, port=config.THRIFT_PORT) as client:
        res = client.send(data)
    return res


def test():
    """pipline to get the test set's scores through thrift server"""
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
    