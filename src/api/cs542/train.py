import numpy as np
import cv2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import callbacks, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras import backend as k
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from pathlib import Path
from pprint import pprint
from easydict import EasyDict as edict
from multiprocessing import cpu_count

from dataset import init_path


k.set_image_dim_ordering('th')
seed = 7
np.random.seed(seed)
num_classes = 2


def load_dataset(path: Path, positive_label: bool=True, length: int=0):
    img_data_list = []
    list_append = img_data_list.append
    i = 0
    for img_path in path.glob('*.jpg'):
        i += 1
        # todo resize
        input_img = cv2.imread(str(img_path))
        input_img = cv2.resize(input_img, (60, 60), interpolation=cv2.INTER_CUBIC)
        input_img = np.swapaxes(input_img, 0, 2)
        list_append(input_img)
    if 0 < length < i:
        img_data_list = shuffle(img_data_list, random_state=2)
        img_data_list = img_data_list[:length]

    img_data = np.array(img_data_list, dtype=np.float32)
    img_data /= 255

    num_of_samples = img_data.shape[0]
    if positive_label:
        labels = np.ones((num_of_samples, ), dtype=np.int64)
        log = {'length of positive': len(labels)}
    else:
        labels = np.zeros((num_of_samples, ), dtype=np.int64)
        log = {'length of negative': len(labels)}
    pprint(log)
    return img_data, labels


def prepare_train_data(dataset_dict: edict):
    labels = np.concatenate((dataset_dict.positive.labels, dataset_dict.negative.labels), axis=0)
    img_data = np.concatenate((dataset_dict.positive.img_data, dataset_dict.negative.img_data), axis=0)
    labels = np_utils.to_categorical(labels, num_classes)
    x, y = shuffle(img_data, labels, random_state=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    return x_train, x_test, y_train, y_test, img_data


def datagen(x_train, y_train, batch_size=128):
    train_steps = int(len(y_train) / batch_size) + 1
    
    train_datagen = ImageDataGenerator(
        rescale=None,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)
    return train_generator, train_steps


class MetricCallback(Callback):
    def __init__(self, predict_batch_size=512, include_on_batch=False):
        super(MetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_end(self, batch, logs=None):
        if self.include_on_batch:
            logs['val_recall'] = np.float32('-inf')
            logs['val_precision'] = np.float32('-inf')
            logs['val_f1'] = np.float32('-inf')
            if self.validation_data:
                y_true = np.argmax(self.validation_data[1], axis=1)
                y_pred = np.argmax(self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size),
                                   axis=1)
                self.set_scores(y_true, y_pred, logs)

    def on_train_begin(self, logs=None):
        if 'val_recall' not in self.params['metrics']:
            self.params['metrics'].append('val_recall')
        if 'val_precision' not in self.params['metrics']:
            self.params['metrics'].append('val_precision')
        if 'val_f1' not in self.params['metrics']:
            self.params['metrics'].append('val_f1')
        if 'val_auc' not in self.params['metrics']:
            self.params['metrics'].append('val_auc')

    def on_epoch_end(self, epoch, logs=None):
        logs['val_recall'] = np.float32('-inf')
        logs['val_precision'] = np.float32('-inf')
        logs['val_f1'] = np.float32('-inf')
        if self.validation_data:
            y_true = np.argmax(self.validation_data[1], axis=1)
            y_pred = np.argmax(self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size), axis=1)
            self.set_scores(y_true, y_pred, logs)
            print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))

    @staticmethod
    def set_scores(y_true, y_pred, logs=None):
        logs['val_recall'] = recall_score(y_true, y_pred)
        logs['val_precision'] = precision_score(y_true, y_pred)
        logs['val_f1'] = f1_score(y_true, y_pred)


def gen_model(input_shape=(3, 30, 30)):
    model = Sequential()
    model.add(Convolution2D(96, 7, 7, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
    return model


def gen_model2(input_shape=(3, 30, 30)):
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(2, (3, 3), padding='same'))
    model.add(GlobalAveragePooling2D(data_format='channels_first'))
    model.add(Activation('softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
    return model


def gen_model3(input_shape=(3, 30, 30)):
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    for layer in conv_base.layers:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    learning_rate = 0.0001
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

    model.summary()
    return model


def train(model, x_train, x_test, y_train, y_test, model_direction, pretrain_model=None):
    csv_log_file = str(Path(model_direction).parent / 'log' / 'model_train_log.csv')
    tensorboard_log_direction = str(Path(model_direction).parent / 'log')
    ckpt_file = str(Path(model_direction) / 'ckpt_model.{epoch:02d}-{val_precision:.4f}.h5')
    # model_file = str(Path(model_direction) / 'latest_model.h5')

    early_stopping = callbacks.EarlyStopping(monitor='val_precision', min_delta=0.0001, patience=50, verbose=0, mode='max')
    csv_log = callbacks.CSVLogger(csv_log_file)
    checkpoint = callbacks.ModelCheckpoint(ckpt_file, monitor='val_precision', verbose=1, save_best_only=True, mode='max')
    tensorboard_callback = callbacks.TensorBoard(log_dir=tensorboard_log_direction, batch_size=32)
    callbacks_list = [MetricCallback(), csv_log, early_stopping, checkpoint, tensorboard_callback]
    if pretrain_model:
        model.load_weights(pretrain_model)
    pprint(model.get_weights()[-1])
    model.fit(x_train, y_train, batch_size=128, epochs=500, verbose=1, validation_data=(x_test, y_test),
              callbacks=callbacks_list)
    # model.save(model_file)
    return model


def train2(model, train_generator, train_steps, x_test, y_test, model_direction, pretrain_model):
    csv_log_file = str(Path(model_direction).parent / 'log' / 'model_train_log.csv')
    tensorboard_log_direction = str(Path(model_direction).parent / 'log')
    ckpt_file = str(Path(model_direction) / 'ckpt_model.{epoch:02d}-{val_precision:.4f}.h5')
    # model_file = str(Path(model_direction) / 'latest_model.h5')

    early_stopping = callbacks.EarlyStopping(monitor='val_precision', min_delta=0.0001, patience=50, verbose=0, mode='max')
    csv_log = callbacks.CSVLogger(csv_log_file)
    checkpoint = callbacks.ModelCheckpoint(ckpt_file, monitor='val_precision', verbose=1, save_best_only=True, mode='max')
    tensorboard_callback = callbacks.TensorBoard(log_dir=tensorboard_log_direction, batch_size=32)
    callbacks_list = [MetricCallback(), csv_log, early_stopping, checkpoint, tensorboard_callback]
    if pretrain_model:
        model.load_weights(pretrain_model)
    pprint(model.get_weights()[-1])

    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=500,
        validation_data=(x_test, y_test),
        verbose=1,
        workers=cpu_count() * 2 + 2,
        use_multiprocessing=True,
        callbacks=callbacks_list)
    return model


def test(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    test_image = x_test[0:1]
    print(test_image.shape)

    print(model.predict(test_image))
    print(model.predict_classes(test_image))
    print(y_test[0:1])


def main():
    blur_directory = '../../../data/output/cs542/train/blurred/'
    clear_directory = '../../../data/output/cs542/train/clear/'
    model_direction = "../../../data/output/cs542/models/"
    pretrain_model = None
    init_path([model_direction])

    img_data_positive, labels_positive = load_dataset(Path(blur_directory), positive_label=True)
    img_data_negative, labels_negative = load_dataset(Path(clear_directory), positive_label=False,
                                                      length=len(labels_positive))
    dataset_dict = edict({
        'positive': {
            'img_data': img_data_positive,
            'labels': labels_positive
        },
        'negative': {
            'img_data': img_data_negative,
            'labels': labels_negative
        }
    })
    x_train, x_test, y_train, y_test, img_data = prepare_train_data(dataset_dict)

    model = gen_model(img_data[0].shape)
    model = train(model, x_train, x_test, y_train, y_test, model_direction, pretrain_model)
    test(model, x_test, y_test)


def main2():
    blur_directory = '../../../data/output/cs542/train/blurred/'
    clear_directory = '../../../data/output/cs542/train/clear/'
    model_direction = "../../../data/output/cs542/models/"
    batch_size = 128
    pretrain_model = None
    init_path([model_direction])


    img_data_positive, labels_positive = load_dataset(Path(blur_directory), positive_label=True)
    img_data_negative, labels_negative = load_dataset(Path(clear_directory), positive_label=False,
                                                      length=len(labels_positive))
    dataset_dict = edict({
        'positive': {
            'img_data': img_data_positive,
            'labels': labels_positive
        },
        'negative': {
            'img_data': img_data_negative,
            'labels': labels_negative
        }
    })
    x_train, x_test, y_train, y_test, img_data = prepare_train_data(dataset_dict)

    model = gen_model3(img_data[0].shape)

    train_generator, train_steps = datagen(x_train, y_train, batch_size)
    print('train_steps: %s' % train_steps)
    model = train2(model, train_generator, train_steps, x_test, y_test, model_direction, pretrain_model)



if __name__ == '__main__':
    # main()
    main2()
    # gen_model3((3, 60, 60))
