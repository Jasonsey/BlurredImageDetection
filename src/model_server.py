# import the necessary packages
import time
import json

import numpy as np
import redis

import config
from db import db
from detection import predict
from utils.server_tools import base64_decode_image


def classify_process():
    while True:
        queue = db.lrange(config.IMAGE_QUEUE, 0, config.BATCH_SIZE - 1)
        data, image_ids = [], []
        for que in queue:
            que = json.loads(que)
            img = base64_decode_image(que['image'], 'float32', que['shape'])
            data.append(img)   
            image_ids.append(que['id'])

        if len(image_ids) > 0:
            print('Batch Size: %s' % len(image_ids))
            scores = predict(data).tolist()
            for (image_id, score) in zip(image_ids, scores):
                result = int(score > 0.5)
                res = {'result': result, 'score': score}
                db.set(image_id, json.dumps(res))
        
            db.ltrim(config.IMAGE_QUEUE, len(image_ids), -1)
        
        time.sleep(settings.SERVER_SLEEP)
            

if __name__ == "__main__":
    classify_process()
