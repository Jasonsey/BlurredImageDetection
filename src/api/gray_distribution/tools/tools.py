import os
from pathlib import Path
from PIL import Image
import cv2
from skimage import filters
import numpy as np
from matplotlib import pyplot as plt


def init_path(paths: list):
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


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def get_imginfo(path, bins=32):
    path = Path(path)
    img = Image.open(path).convert('L')
    
    area = img.height * img.width

    img_array = np.array(img).flatten()
    # plt.hist(img_array.flatten(), bins=bins)
    # plt.show()
    hist_array = []
    start = 0
    stop = 256
    if (stop-start) % bins:
        stop += bins - (stop-start) % bins
    step = (stop - start) // bins
    for i in range(start, stop, step):
        print(i, start, stop, step)
        score = ((img_array>i)[img_array<i+step]).sum()
        hist_array.append(score)
    hist_array = np.array(hist_array) / area
    return hist_array


if __name__ == '__main__':
    print('Debug~!')
    for p in Path('../../../data/input/License/Train/Bad_License/').glob('*.jpg'):
        get_imginfo(p)
        break
