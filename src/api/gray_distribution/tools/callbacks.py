import sys
import numpy as np
from pathlib import Path
from keras.callbacks import Callback, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from keras.utils import np_utils

sys.path.append('..')
from config import Config
from tools.tools import init_path


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
                y_true = self.validation_data[1]
                y_pred = self.model.predict_classes(self.validation_data[0], batch_size=self.predict_batch_size)
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
            y_true = self.validation_data[1]
            y_pred = self.model.predict_classes(self.validation_data[0], batch_size=self.predict_batch_size)
            self.set_scores(y_true, y_pred, logs)
            print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))

    @staticmethod
    def set_scores(y_true, y_pred, logs=None):
        logs['val_recall'] = recall_score(y_true, y_pred)
        logs['val_precision'] = precision_score(y_true, y_pred)
        logs['val_f1'] = f1_score(y_true, y_pred)


def callbacks(output_path):
    output_path = Path(output_path)

    csv_log_file = str(output_path / 'log' / 'model_train_log.csv')
    tensorboard_log_direction = str(output_path / 'log')
    model_direction = output_path / 'models'
    init_path([model_direction])
    ckpt_file = str(model_direction / 'ckpt_model.{epoch:02d}-{val_acc:.4f}.h5')

    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=200, verbose=0, mode='max')
    csv_log = CSVLogger(csv_log_file)
    checkpoint = ModelCheckpoint(ckpt_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_direction, batch_size=256)
    callbacks_list = [MetricCallback(), csv_log, early_stopping, checkpoint, tensorboard_callback]
    return callbacks_list

