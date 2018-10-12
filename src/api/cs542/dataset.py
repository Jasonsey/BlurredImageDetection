# 使用现有的清晰、模糊图片创建训练数据集
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from tools import resize
from tools import xml_dataset


grid_x = 30
grid_y = 30


def init_path(paths: list):
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


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


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def crop_images(goods: list, bads: list, input_path: Path, clear_path: Path, blur_path: Path):
    images = [goods, bads]
    for i in range(len(images)):
        for ii in range(len(images[i])):
            img = images[i][ii]
            # img = focuse_image(img)
            # img = resize(img, size_max=360)
            range_x = img.width // grid_x
            range_y = img.height // grid_y
            for x in range(range_x):
                for y in range(range_y):
                    bbox = (x * grid_x, y * grid_y, x * grid_x + grid_x, y * grid_y + grid_y)
                    slice_bit = img.crop(bbox)
                    if i == 0:
                        path1 = Path(input_path) / ('clear_' + str(ii) + '_' + str(x) + '_' + str(y) + '.jpg')
                        path2 = Path(clear_path) / ('clear_' + str(ii) + '_' + str(x) + '_' + str(y) + '.jpg')
                    else:
                        path1 = Path(input_path) / ('blur_' + str(ii) + '_' + str(x) + '_' + str(y) + '.jpg')
                        path2 = Path(blur_path) / ('blur_' + str(ii) + '_' + str(x) + '_' + str(y) + '.jpg')
                    slice_bit.save(path1, optimize=True)
                    slice_bit.save(path2, optimize=True)
            print(ii)
    print('All Done !')


def main():
    # ori_path = "../../../data/input/License/Train"
    xml_path = '../../../data/input/License/Block_License/Seleted'
    img_path = '../../../data/input/License/Block_License'

    clear_path = "../../../data/output/cs542/train/clear/"
    blur_path = "../../../data/output/cs542/train/blurred/"
    input_path = "../../../data/output/cs542/train/input_data/"

    init_path([clear_path, blur_path, input_path])

    # good_img, bad_img = load_images(Path(ori_path))
    good_img, bad_img = xml_dataset(xml_path, img_path)
    crop_images(good_img, bad_img, Path(input_path), Path(clear_path), Path(blur_path))


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
