import cv2
from PIL import Image
from pathlib import Path
import numpy as np
from pprint import pprint
from keras import backend as k
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix
import struct

from train import gen_model2 as gen_model
from tools import resize


k.set_image_dim_ordering('th')
input_path = Path('../../../data/input/License/Train')
output_path = Path('../../../data/output/cs542/output')
if not output_path.exists():
    output_path.mkdir(parents=True)


def focuse_image(img):
    w, h = 4, 3
    width, height = img.width, img.height
    w, h = (width // w, height // h) if width < height else (width // h, height // w)
    box = (w, h, width - w, height - h)
    return img.crop(box)


def split_image(path):
    print(path)
    gridx, gridy = 30, 30
    img = Image.open(path)
    img = img.convert('RGB')
    img, _ = resize(img)
    img = focuse_image(img)     # focus img to center
    rangex, rangey = img.width // gridx, img.height // gridy

    img_data_list = []
    list_append = img_data_list.append
    for x in range(rangex):
        for y in range(rangey):
            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = np.asarray(img.crop(bbox))
            slice_bit = cv2.cvtColor(slice_bit, cv2.COLOR_RGB2BGR)
            slice_bit = np.swapaxes(slice_bit, 0, 2)
            list_append(slice_bit)
    img_array = np.array(img_data_list)
    img_array = img_array.astype(np.float32)
    img_array /= 255
    print(img_array.shape)
    return img_array, rangex, rangey


def count_array(array, rangex, rangey):
    data = array[:, 1].reshape((rangex, rangey))
    p = (array[:, 1] >= 0.5).sum()
    n = (array[:, 1] < 0.5).sum()
    score = float(p) / (p + n)
    # mean = np.mean(array, axis=0)
    # score = mean[1]
    print(score, score > 0.5)
    # positive = array.sum()
    # negative = len(array) - positive
    # score = float(positive) / (negative + positive)
    # print(positive, negative, positive > negative)
    # return score, positive > negative
    return score, score > 0.5, data


def best_model(model_path: Path):
    best_precision = 0
    best_path = 'latest_model.h5'
    for path in model_path.glob('*.h5'):
        if 'ckpt_model' in path.stem:
            # print(path.stem.split('-'))
            val_precision = float(path.stem.split('-')[1])
            if val_precision > best_precision:
                best_precision = val_precision
                best_path = path
    return str(best_path)


def predict():
    paths_list = [input_path.glob('**/*.jpg')]
    model = gen_model(input_shape=(3, 30, 30))
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model_path = best_model(Path('../../../data/output/cs542/models'))
    model.load_weights(model_path)
    # model.load_weights('../../../data/output/cs542/models/latest_model.h5')
    pprint(model.get_weights()[-1])

    good_output = output_path / 'Good_Images'
    bad_output = output_path / 'Bad_Images'
    re_output = output_path / 'Re_Images'

    for p in [good_output, bad_output, re_output]:
        if not p.exists():
            p.mkdir(parents=True)

    for paths in paths_list:
        for path in paths:
            try:
                img_array, rangex, rangey = split_image(path)
            except (struct.error, OSError) as e:
                print('Error: %s, path: %s' % (e, path))
                continue
            score, flag, data = count_array(model.predict(img_array), rangex, rangey)
            img = Image.open(path)
            # img = img.convert('RGB')
            df = DataFrame(data)
            if path.parent.name == 'Good':
                img.save(str(good_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(good_output / ('{:.4f}.csv'.format(score))))
            elif path.parent.name == 'Bad':
                img.save(str(bad_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(bad_output / ('{:.4f}.csv'.format(score))))
            else:
                img.save(str(re_output / ('{:.4f}-{}.jpg'.format(score, path.stem))))
                # df.to_csv(str(re_output / ('{:.4f}.csv'.format(score))))


def test():
    paths_list = [input_path.glob('**/*.jpg')]
    model = gen_model(input_shape=(3, 30, 30))
    pprint(model.trainable_weights)
    pprint(model.get_weights()[-1])
    model_path = best_model(Path('../../../data/output/cs542/models'))
    model.load_weights(model_path)
    # model.load_weights('../../../data/output/cs542/models/latest_model.h5')
    pprint(model.get_weights()[-1])

    y_true, y_pred = [], []
    for paths in paths_list:
        for path in paths:
            try:
                img_array, rangex, rangey = split_image(path)
            except (struct.error, OSError) as e:
                print('Error: %s, path: %s' % (e, path))
                continue
            if path.parent.name == 'Bad_License':
                y_true.append(1)
            elif path.parent.name == 'Good_License':
                y_true.append(0)
            else:
                # continue
                y_true.append(0)
            score, flag, data = count_array(model.predict(img_array), rangex, rangey)
            if score > 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
    print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    # predict()
    test()
    # try:
    #     split_image('../../../data/input/License/Test7/6732.jpg')
    # except (struct.error, OSError):
    #     print('error')
