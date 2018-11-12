import json
import io

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer, TProcessPoolServer
from thrift import Thrift
import numpy as np
from PIL import Image

from api.thrift_api.interface import blur_detection
from detection import predict
import config


class InterfaceHandler(object):
    def reco(self, imgbytes):
        print('\nserver process...')
        with io.BytesIO(imgbytes) as f:
            img = Image.open(f)
            img = self.format_img(img)
            img_array = np.asarray(img)[np.newaxis, :]
            print(img_array.shape)
        score = float(predict(img_array)[0])

        pred = 1 if score > 0.5 else 0
        res = {
            'data': {
                'score': score,
                'pred': pred, 
            },
            'status': 1,
            'msg': ''
        }
        return json.dumps(res)
    
    def format_img(self, image):
        '''部分图片直接转换成RGB，会出现颜色反转现象'''
        if image.mode == 'RGBA':
            r, g, b, a = image.split()
            rs = np.array(r)
            rs = (rs < 1)
            num = rs.sum()
            if num * 2 > image.width * image.height:
                imgbg = Image.new(image.mode, image.size, (255, 255, 255, 255))
                imgbg.paste(image, a)
                image = imgbg
        image = image.convert('RGB')
        return image


def main():
    # TODO 服务端多进程被调用时，速度并没有很大提升，待解决
    # 创建服务端
    handler = InterfaceHandler()
    processor = blur_detection.Processor(handler)
    # 监听端口
    transport = TSocket.TServerSocket("0.0.0.0", config.THRIFT_PORT)
    # 选择传输层
    tfactory = TTransport.TBufferedTransportFactory()
    # 选择传输协议
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    # 创建服务端
    # server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    # server.setNumThreads(12)
    server = TProcessPoolServer.TProcessPoolServer(processor, transport, tfactory, pfactory)
    server.setNumWorkers(config.THRIFT_NUM_WORKS)
    print("Starting thrift server in python...")
    server.serve()
    print("done!")


if __name__ == '__main__':
    main()


