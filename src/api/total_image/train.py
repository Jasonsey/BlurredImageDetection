"""
train.py
"""

from pathlib import Path
from multiprocessing import cpu_count

from easydict import EasyDict
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from utils.callbacks import get_callbacks
from utils.tools import init_path
from dataset import read_dataset
from .model import MODEL
import config


def train(model, input_dataset: EasyDict, model_direction, pretrain_model):
    # # config TF Session
    # config = tf.ConfigProto()  
    # config.gpu_options.allow_growth=True   
    # session = tf.Session(config=config)
    # set_session(session)

    callbacks_list = get_callbacks(
        model_direction,
        epochsize=input_dataset.epoch_size,
        batchsize=input_dataset.batch_size
    )
    if pretrain_model:
        model.load_weights(pretrain_model)

    model.fit_generator(
        input_dataset.train,
        steps_per_epoch=input_dataset.train_steps,
        epochs=1000,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        verbose=1,
        workers=cpu_count(),
        use_multiprocessing=True,
        max_queue_size=192,
        callbacks=callbacks_list)
    return model


def main():
    blur_path = '../data/input/License/Train/Bad_License/'
    clear_path = '../data/input/License/Train/Good_License/'
    model_direction = '../data/output/total_image/models/'
    batch_size = config.BATCH_SIZE
    pretrain_model = None
    init_path([model_direction])

    input_dataset = read_dataset.read_dataset2(
        paths=[clear_path, blur_path],
        batch_size=batch_size
    )

    model = MODEL(input_shape=input_dataset.input_shape)
    model = train(model, input_dataset, model_direction, pretrain_model)


if __name__ == '__main__':
    # main()
    main2()
