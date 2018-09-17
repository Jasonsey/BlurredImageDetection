# 使用现有的清晰、模糊图片创建训练数据集
from PIL import Image
from pathlib import Path


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


def crop_images(goods: list, bads: list, input_path: Path, clear_path: Path, blur_path: Path):
    images = [goods, bads]
    for i in len(images):
        for ii in len(images[i]):
            img = images[i][ii]
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
                    slice_bit.save(path1, optimize=True, bits=6)
                    slice_bit.save(path2, optimize=True, bits=6)


def main():
    ori_path = "../../../data/input/License/Train"

    clear_path = "../../../data/output/cs542/train/clear/"
    blur_path = "../../../data/output/cs542/train/blurred/"
    input_path = "../../../data/output/cs542/train/input_data/"

    init_path([clear_path, blur_path, input_path])

    good_img, bad_img = load_images(Path(ori_path))
    crop_images(good_img, bad_img, Path(input_path), Path(clear_path), Path(blur_path))


if __name__ == '__main__':
    main()
