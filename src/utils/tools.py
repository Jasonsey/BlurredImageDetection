import os
from pathlib import Path
from PIL import Image
import cv2
from skimage import filters
import numpy as np


def init_path(paths: list):
    '''
    创建路径，不存在则新建
    '''
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


def resize(im: Image, size_min=1300, reduce_anything=True):
    """
    resize image.
    """
    width, height = im.width, im.height
    if width > size_min and height > size_min and not reduce_anything:
        return im, 1.0
    if width > height:
        ratio = float(size_min) / height
    else:
        ratio = float(size_min) / width
    width, height = int(width * ratio), int(height * ratio)
    im = im.resize((width, height), Image.ANTIALIAS)
    return im, ratio


def resize2(im: Image, size=256):
    """
    调整大小到(x,x)尺寸
    """
    im = im.resize((size, size), Image.ANTIALIAS)
    return im


def focuse_image(img):
    '''
    聚焦到图片中心
    '''
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def get_imginfo(path):
    '''
    非协程版info读取接口, 返回(tenengrad_score, laplacian_score, area)
    '''
    path = Path(path)
    img = Image.open(path).convert('L')

    area = img.height * img.width

    img = img.resize((800, 800), Image.ANTIALIAS)
    img_array = np.array(img)
    temp = filters.sobel(img_array)
    tenengrad_score = temp.var()  # sobel score

    img = cv2.imread(str(path), 0)
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
    laplacian_score = cv2.Laplacian(img, cv2.CV_64F).var()
    return tenengrad_score, laplacian_score, area


async def get_imginfo2(path):
    '''
    协程版info读取接口
    '''
    return get_imginfo(path)


async def get_imgarray(path):
    '''
    协程读取图片到array
    '''
    img = Image.open(path).convert('RGB')
    img = focuse_image(img)
    img = resize2(img)
    return np.asarray(img)


def get_imginfo_by_array(array):
    img = Image.fromarray(array).convert('L')
    cv2_img = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    area = img.height * img.width

    img = img.resize((800, 800), Image.ANTIALIAS)
    img_array = np.array(img)
    temp = filters.sobel(img_array)
    tenengrad_score = temp.var()  # sobel score

    img = cv2_img
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
    laplacian_score = cv2.Laplacian(img, cv2.CV_64F).var()
    return tenengrad_score, laplacian_score, area



if __name__ == '__main__':
    a = []
    for p in Path('data/input/License/temp/Bad_License/').glob('*.jpg'):
        a.append(list(get_imginfo(p)))
    ar = np.array(a)
    print(ar.shape, ar.mean(axis=0), ar.std(axis=0))
    ar = (ar-ar.mean(axis=0)) / ar.std(axis=0)
    print(ar[:3])
    