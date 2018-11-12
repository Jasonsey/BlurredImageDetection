import sys 
import asyncio
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from easydict import EasyDict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

from utils.tools import get_imginfo, get_imginfo2, get_imgarray, focuse_image, resize2, init_path
import config 
from api.decision_tree import detection as tree_detection
from api.total_image import detection as cnn_detection


def load_dataset(paths: list, random=True):
    '''
    用于decision_tree使用的读取数据接口
    '''
    data, labels = [], []
    for i in range(len(paths)):
        path = Path(paths[i])
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for pa in path.glob('**/*.jpg'):
            print(pa)
            results.append(asyncio.ensure_future(get_imginfo2(pa)))
            labels.append(i)
        loop.run_until_complete(asyncio.wait(results))
        for result in results:
            data.append(list(result.result()))

    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='float32')
    print('Blur: %s, Clear: %s' % ((labels==1).sum(), (labels==0).sum()))
    if random:
        data, labels = shuffle(data, labels, random_state=2) 
    return data, labels


def load_dataset2(paths: list, random=True):
    '''
    加载total_image数据集
    '''
    data, labels = [], []
    for i in range(len(paths)):
        path = Path(paths[i])
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for pa in path.glob('*.jpg'):
            print(pa)
            results.append(asyncio.ensure_future(get_imgarray(pa)))
            labels.append(i)
        loop.run_until_complete(asyncio.wait(results))
        for result in results:
            data.append(result.result())

    data = np.asarray(data, dtype='float32')
    data /= 255
    labels = np.asarray(labels, dtype='float32')
    print('Blur: %s, Clear: %s' % ((labels==1).sum(), (labels==0).sum()))
    if random:
       data, labels = shuffle(data, labels, random_state=2) 
    return data, labels


def load_dataset3(paths: list, random=True):
    '''
    从本地硬盘加载stacking的数据集, 返回tree + cnn得分，以及原始的label
    '''
    data_info, data_array, labels = [], [], []
    for i in range(len(paths)):
        path = Path(paths[i])
        results = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for pa in path.glob('*.jpg'):
            print(pa)
            results.append(asyncio.ensure_future(imginfo_and_array(pa)))
            labels.append(i)
        loop.run_until_complete(asyncio.wait(results))
        for result in results:
            imginfo, array = result.result()
            data_info.append(imginfo)
            data_array.append(array)
    labels = np.asarray(labels, dtype='float32')
    data_info = np.asarray(data_info, dtype='float32')
    data_info = tree_detection.predict(data_info)
    data_info = data_info[:, 1].reshape((-1, 1))

    data_array = np.asarray(data_array, dtype='float32')
    data_array = cnn_detection.predict(data_array)
    data_array = data_array
    
    data = np.concatenate((data_info, data_array), axis=1)
    if random:
        data, labels = shuffle(data, labels, random_state=2) 
    return data, labels


async def imginfo_and_array(path):
    img = Image.open(path).convert('RGB')
    imginfo = list(get_imginfo(path))

    img = focuse_image(img)
    img = resize2(img)
    array = np.asarray(img, dtype='float32')
    return imginfo, array


def split_dataset(*array):
    return train_test_split(*array, test_size=0.2, random_state=2)


def datagen(x_train, y_train, batch_size=128):
    epoch_size = len(y_train)
    if epoch_size % batch_size < batch_size / config.GPUS:    # 使用多GPU时，可能出现其中1个GPU 0 batchsize问题
        x_train = x_train[:-(epoch_size % batch_size)]
        y_train = y_train[:-(epoch_size % batch_size)]
    epoch_size = len(y_train)
    if epoch_size % batch_size:
        train_steps = int(epoch_size / batch_size) + 1
    else:
        train_steps = int(epoch_size / batch_size)
        
    train_datagen = ImageDataGenerator(
        rescale=None,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)
    return train_generator, train_steps, epoch_size


def read_dataset(paths: list, use_cache=True, cache_home='../data/output/cache'):
    '''
    decision_tree使用的数据集读取接口
    '''
    cache_home = Path(cache_home)
    init_path([cache_home])
    cache_path = Path(cache_home) / 'dataset_decision_tree.pkl'
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        assert len(paths) == config.NUM_CLASS, 'length of paths should be %s, but get %s' % (NUM_CLASS, len(paths))

        data, labels = load_dataset(paths)
        x_train, x_test, y_train, y_test = split_dataset(data, labels)

        dataset = EasyDict({
            'train': {
                'data': x_train,
                'labels': y_train
            },
            'test': {
                'data': x_test,
                'labels': y_test
            }
        })
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    print('All Dataset Read!')
    return dataset


def read_dataset2(paths: list, batch_size=128, use_cache=True, cache_home='../data/output/cache'):
    '''
    读取total_image图片训练数据集
    '''
    cache_home = Path(cache_home)
    init_path([cache_home])
    cache_path = Path(cache_home) / 'dataset_total_image.pkl'
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache_dataset = pickle.load(f)
        x_train = cache_dataset.train.data
        y_train = cache_dataset.train.labels
        x_test = cache_dataset.test.data
        y_test = cache_dataset.test.labels
    else:
        assert len(paths) == config.NUM_CLASS, 'length of paths should be %s, but get %s' % (NUM_CLASS, len(paths))
        
        data, labels = load_dataset2(paths)
        x_train, x_test, y_train, y_test = split_dataset(data, labels)
        cache_dataset = EasyDict({
            'train': {
                'data': x_train,
                'labels': y_train
            },
            'test': {
                'data': x_test,
                'labels': y_test
            }
        })
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_dataset, f)
    
    train_generator, train_steps, epoch_size = datagen(x_train, y_train, batch_size)
    dataset = EasyDict({
        'train': train_generator, 
        'test': {
            'data': x_test,
            'labels': y_test
        },
        'train_steps': train_steps,
        'epoch_size': epoch_size,
        'input_shape': x_train[0].shape,
        'batch_size': batch_size
    })
    print('All Dataset Read!')
    return dataset


def read_dataset3(paths: list, use_cache=True, cache_home='../data/output/cache'):
    '''
    读取stacking的数据集
    '''
    cache_home = Path(cache_home)
    init_path([cache_home])
    cache_path = Path(cache_home) / 'dataset_stacking.pkl'
    if use_cache and cache_path.exists():
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        assert len(paths) == config.NUM_CLASS, 'length of paths should be %s, but get %s' % (NUM_CLASS, len(paths))
        data, labels = load_dataset3(paths)

        x_train, x_test, y_train, y_test = split_dataset(data, labels)

        dataset = EasyDict({
            'train': {
                'data': x_train,
                'labels': y_train
            },
            'test': {
                'data': x_test,
                'labels': y_test
            }
        })
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

    print('All Dataset Read!')
    return dataset


if __name__ == '__main__':
    pass
