"""
check_dataset
"""

from pathlib import Path
from pprint import pprint
from PIL import Image
from tools.tools import init_path


GRID_X = 30
GRID_Y = 30


def gen_pics_dict(input_path: Path, verbose: int=0):
    pics_dict = {}
    for path in input_path.glob('*.jpg'):
        stem = path.stem.split('_')
        if stem[1] in pics_dict:
            pics_dict[stem[1]].append(path)
        else:
            pics_dict[stem[1]] = [path]
    if verbose:
        print('Generated the pics dictionary')
    return pics_dict


def pic_size(paths: list):
    """
    Enter a path list for all small images of an image
    """
    x_max, y_max = 0, 0
    for path in paths:
        stem = path.stem.split('_')
        x, y = int(stem[2]), int(stem[3])
        x_max = x if x > x_max else x_max
        y_max = y if y > y_max else y_max
    width = x_max * GRID_X +GRID_X 
    height = y_max * GRID_Y +GRID_Y 
    pprint({'height': height, 'width': width})
    return width, height


def merge_pics(pics_dict: dict, output_path: Path, verbose: int=0):
    for k in pics_dict:
        paths = pics_dict[k]
        width, height = pic_size(paths)
        image = Image.new(mode='RGB', size=(width, height), color=(255, 255, 255))
        img_path = output_path / '{}.jpg'.format(k)
        for path in paths:
            _, _, x, y = path.stem.split('_')
            x, y = int(x), int(y)
            box = (x * GRID_X, y * GRID_Y, x * GRID_X + GRID_X, y * GRID_Y + GRID_Y)
            slice_bit = Image.open(path)
            image.paste(slice_bit, box)
        if verbose:
            print('Merge new pic: {}'.format(str(img_path)))
        image.save(img_path)
    print('All pics are merged. The path is {}'.format(str(output_path)))


def main():
    input_path = Path("../../../data/output/cs542/train/input_data/")
    check_input_path = Path('../../../data/output/cs542/train/check_input/')
    init_path([check_input_path])

    pics_dict = gen_pics_dict(input_path)
    merge_pics(pics_dict, check_input_path, verbose=1)


if __name__ == '__main__':
    main()
