from pathlib import Path
from easydict import EasyDict

from tools.tools import init_path
from dataset.read_dataset import read_dataset
from model import MODEL
from tools.callbacks import callbacks


def train(model, input_dataset: EasyDict, output_path, pretrain=None):
    callbacks_list = callbacks(output_path)
    if pretrain is not None:
        model.load_weights(pretrain)

    model.fit(
        input_dataset.train.data,
        input_dataset.train.labels,
        batch_size=128,
        epochs=500,
        verbose=1,
        validation_data=(input_dataset.test.data, input_dataset.test.labels),
        callbacks=callbacks_list)
    print(model.get_weights()[0])
    return model


def main():
    blur_path = '../../../data/input/License/Train2/Bad_License/'
    clear_path = '../../../data/input/License/Train2/Good_License/'
    output_path = '../../../data/output/gray_distribution/'
    pretrain = None

    # init_path([output_path])

    x_train, y_train, x_test, y_test = read_dataset([clear_path, blur_path])
    input_dataset = EasyDict({
        'train': {
            'data': x_train,
            'labels': y_train},
        'test': {
            'data': x_test,
            'labels': y_test}})
    model = MODEL(input_shape=x_train[0].shape)

    model = train(model, input_dataset, output_path)


if __name__ == '__main__':
    main()
