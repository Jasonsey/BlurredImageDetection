# 使用正常的图片创建模糊图片数据集
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path
import cv2
import numpy as np
from random import randint
from pprint import pprint

grid_x = 30
grid_y = 30
grid_w = 20
DEBUG = False


def init_path(paths: list):
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


class Blur(object):
    """
    functions for blur images
    """
    def __init__(self, img: Image):
        self._box_size = 25
        self._radius = 2
        self._degree = 12
        self._angle = 45
        self.image = img.copy()
        self.width = self.image.width
        self.height = self.image.height

    def motion(self, degree=None, angle=None):
        degree = self._degree if degree is None else degree
        angle = self._angle if angle is None else angle
        image = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)       # PIL -> cv2

        m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        self.image = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))      # cv2 -> PIL
        return self

    def mosaic(self, box_size=None):
        """
        mosaic pics
        :param box_size: int
        :return: None
        """
        image = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        if box_size is None:
            box_size = self._box_size
        for x in range(0, self.width, box_size):
            for y in range(0, self.height, box_size):
                r, g, b = self.image.getpixel((x, y))
                draw.rectangle(((x, y), (x + box_size, y + box_size)), fill=(r, g, b), outline=None)
        self.image = image
        return self

    def gaussian(self, radius=None):
        if radius is None:
            radius = self._radius
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius=radius))
        return self

    def var(self):
        return self.image


def gen_parameter():
    """
    if box_size < 3 and degree < 6 and radius < 2: pics are clear, so the function stop this possibility
    @ Return:
    -----------
    box_size: 1-2, pics is clear, others are fuzzy
    degree: 1-5, pic is clear, others are fuzzy
    radius: 0-1, pic is clear, others are fuzzy
    angle: 0-360, the motion blur angle
    """
    box_threshold, degree_threshold, radius_threshold = 3, 6, 2
    while True:
        box_size, degree, radius = randint(1, 4), randint(1, 15), randint(0, 4)
        if not (box_size < box_threshold and degree < degree_threshold and radius < radius_threshold):
            break
    angle = randint(0, 359)
    if DEBUG:
        pprint({'box_size': box_size, 'degree': degree, 'angle': angle, 'radius': radius})
    return box_size, degree, angle, radius


def load_images(ori_path: Path):
    images = []
    print('begin loading images')
    for image in ori_path.glob('*.jpg'):
        images.append(Image.open(image).convert('RGB'))
    print('finished loading images')
    return images


def resize(im: Image, size_max=900):
    """
    resize image. If the longest side of the image exceeds size_max, the image will be reduced to the longest side
    without exceeding size_max, otherwise it will not be reduced
    """
    width, height = im.width, im.height
    if width < 900 and height < 900:
        return im
    if width > height:
        ratio = float(width) / size_max
    else:
        ratio = float(height) / size_max
    width, height = int(width * ratio), int(height * ratio)
    im = im.resize((width, height), Image.ANTIALIAS)
    return im



def crop_images(images: list, input_path: Path, clear_path: Path, blur_path: Path):
    """
    Cut the image into small pieces, then randomly process some small pieces into blurred pictures
    """
    for i in range(len(images)):
        print(i)
        img = images[i]
        range_x = int(img.width / grid_x)
        range_y = int(img.height / grid_y)
        for x in range(range_x):
            for y in range(range_y):
                bbox = (x * grid_x, y * grid_y, x * grid_x + grid_x, y * grid_y + grid_y)
                slice_bit = img.crop(bbox)
                flag = randint(0, 1)        # random save to different folder
                if flag:
                    path1 = Path(input_path) / ('clear_' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg')
                    path2 = Path(clear_path) / ('clear_' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg')
                else:
                    box_size, degree, angle, radius = gen_parameter()
                    blur = Blur(slice_bit)
                    slice_bit = blur.mosaic(box_size=box_size).motion(degree=degree, angle=angle)\
                        .gaussian(radius=radius).var()

                    path1 = Path(input_path) / ('blur_' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg')
                    path2 = Path(blur_path) / ('blur_' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg')
                slice_bit.save(path1, optimize=True, bits=6)
                slice_bit.save(path2, optimize=True, bits=6)


def main():
    ori_path = "../../../data/input/License/All_Good_Images"

    clear_path = "../../../data/output/cs542/train/clear/"
    blur_path = "../../../data/output/cs542/train/blurred/"
    input_path = "../../../data/output/cs542/train/input_data/"

    init_path([clear_path, blur_path, input_path])
    images = load_images(Path(ori_path))
    crop_images(images, Path(input_path), Path(clear_path), Path(blur_path))


if __name__ == '__main__':
    # main()
    # path = Path('../../../data/input/License/Train/Good_License/23-115730.jpg')
    # blur = Blur(path)
    # blur.var().show()
    # blur.mosaic(box_size=2).motion(degree=6, angle=0).gaussian(radius=1).var().show()
    # blur.motion(degree=1, angle=0).gaussian(radius=4).var().show()
    # gen_parameter()
    p = "../../../data/input/License/Small_Good_Images"
    imgs = load_images(Path(p))
    print(len(imgs))
    for img in imgs:
        image = Image.new('RGB', (img.width, img.height), color=(255, 255, 255))

        range_x = img.width // grid_x
        range_y = img.height // grid_y
        for x in range(range_x):
            for y in range(range_y):
                box = (x * grid_x, y * grid_y, x * grid_x + grid_x, y * grid_y + grid_y)

                if randint(0, 1):
                    slice_bit = img.crop(box)
                    image.paste(slice_bit, box)
                else:
                    img_copy = img.copy()
                    big_box = (box[0] - grid_w, box[1] - grid_w, box[2] + grid_w, box[3] + grid_w)
                    big_slice = img.crop(big_box)
                    box_size, degree, angle, radius = gen_parameter()
                    print(box_size, degree, angle, radius)
                    blur = Blur(big_slice)
                    big_slice = blur.mosaic(box_size=box_size).motion(degree=degree, angle=angle)\
                        .gaussian(radius=radius).var()
                    # big_slice = blur.motion(degree=6, angle=angle)\
                    #     .var()
                    # big_slice.show()
                    slice_bit = big_slice.crop((grid_w, grid_w, grid_w + grid_x, grid_w + grid_y))
                    image.paste(slice_bit, box)
        img.show()
        image.show()
        input('OK?:')


