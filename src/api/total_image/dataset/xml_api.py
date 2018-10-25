import sys
from pathlib import Path
from xml.dom import minidom
from PIL import Image

sys.path.append('..')
from tools.tools import resize


def read_xml(xml_path):
    xml_path = Path(xml_path)
    for xml in xml_path.glob('**/*.xml'):
        # read xml document
        dom = minidom.parse(str(xml))
        annotation = dom.documentElement
        yield annotation


def read_img(annotations, img_path):
    img_path = Path(img_path)
    clear_imgs = []
    blur_imgs = []
    for annotation in annotations:
        folder = annotation.getElementsByTagName('folder')[0].firstChild.data
        filename = annotation.getElementsByTagName('filename')[0].firstChild.data
        img = Image.open(img_path / folder / filename)
        img, ratio = resize(img)

        for obj in annotation.getElementsByTagName('object'):
            name = obj.getElementsByTagName('name')[0].firstChild.data
            xmin = int(obj.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(obj.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(obj.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(obj.getElementsByTagName('ymax')[0].firstChild.data)
            box = [int(xmin*ratio), int(ymin*ratio), int(xmax*ratio), int(ymax*ratio)]
            slice_img = img.crop(box)
            slice_img, _ = resize(slice_img, size_min=30, reduce_anything=False)

            if name == 'blur':
                blur_imgs.append(slice_img)
            elif name == 'clear':
                clear_imgs.append(slice_img)
            else:
                pass
    return clear_imgs, blur_imgs



def xml_dataset(xml_path, img_path):
    annotations = read_xml(xml_path)
    return read_img(annotations, img_path)


def debug():
    xml_path = '../../../data/input/License/Block_License/Seleted'
    img_path = '../../../data/input/License/Block_License'

    annotations = read_xml(xml_path)
    clears, blurs = read_img(annotations, img_path)
    for i in blurs:
        i.show()
        input('...')


if __name__ == '__main__':
    debug()
