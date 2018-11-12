import cv2
import asyncio
import numpy as np


def predict(arrays):
    '''
    opencv laplacian port
    '''
    async def get_score(array):
        img = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (500, 500))  # 为方便与其他图片比较可以将图片resize到同一个大小
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        return score

    results = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for array in arrays:
        results.append(asyncio.ensure_future(get_score(array)))
    loop.run_until_complete(asyncio.wait(results))

    scores = []
    for result in results:
        score = result.result()
        scores.append(score)
    return np.asarray(scores)


if __name__ == '__main__':
    pass