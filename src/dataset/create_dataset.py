"""
使用现有的清晰、模糊图片创建训练数据集
"""

import sys
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

from utils.tools import init_path, resize2, focuse_image



def load_images(ori_path1: Path):
    goods = []
    bads = []
    paths = [ori_path1 / 'Good_License', ori_path1 / 'Bad_License']

    print('begin loading images')
    for path in paths:
        for image in path.glob('*.jpg'):
            if path.name == 'Good_License':
                goods.append(Image.open(image).convert('RGB'))
            else:
                bads.append(Image.open(image).convert('RGB'))
    print('finished loading images')
    return goods, bads


def crop_images(goods: list, bads: list, clear_path: Path, blur_path: Path):
    images = [goods, bads]
    for i in range(len(images)):
        for ii in range(len(images[i])):
            img = images[i][ii]
            img = focuse_image(img)
            img = resize2(img, size=256)
            if i == 0:
                path = Path(clear_path) / ('clear_' + str(ii) + '.jpg')
            else:
                path = Path(blur_path) / ('blur_' + str(ii) + '.jpg')
            img.save(path, optimize=True)
            print(ii)
    print('All Done !')


def main():
    ori_path = "../data/input/License/Train"
    clear_path = "../data/output/total_image/train/clear/"
    blur_path = "../data/output/total_image/train/blurred/"

    init_path([clear_path, blur_path])

    good_img, bad_img = load_images(Path(ori_path))
    crop_images(good_img, bad_img, Path(clear_path), Path(blur_path))


def debug():
    ori_path = "../../../data/input/License/temp"
    good_img, bad_img = load_images(Path(ori_path))
    for img in good_img:
        im = focuse_image(img)
        im = resize(im, size_min=340)
        plt.imshow(im)
        plt.show()
        # img.show()
        # im.show()
        input('....')


if __name__ == '__main__':
    main()
    # debug()
