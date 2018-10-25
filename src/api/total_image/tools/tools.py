from pathlib import Path
from PIL import Image


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


def resize2(im: Image, size=256):
    """
    resize image
    """
    im = im.resize((size, size), Image.ANTIALIAS)
    return im


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


if __name__ == '__main__':
    pass
