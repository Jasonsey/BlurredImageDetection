from pathlib import Path
import numpy as np
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


class MetricCallback(Callback):
    def __init__(self, predict_batch_size=64, include_on_batch=False):
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
            y_true = self.validation_data[1]
            y_ = self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)
            # y_.shape = (None, 1)
            y_pred = (y_>0.5).astype('float32').flatten()
            self.set_scores(y_true, y_pred, logs)
            print(classification_report(y_true, y_pred, target_names=['清晰', '模糊']))

    @staticmethod
    def set_scores(y_true, y_pred, logs=None):
        logs['val_recall'] = recall_score(y_true, y_pred)
        logs['val_precision'] = precision_score(y_true, y_pred)
        logs['val_f1'] = f1_score(y_true, y_pred)


class SGDRScheduler(Callback):
    '''Schedule learning rates with restarts
    A simple restart technique for stochastic gradient descent.
    The learning rate decays after each batch and peridically resets to its
    initial value. Optionally, the learning rate is additionally reduced by a
￼    fixed factor at a predifined set of epochs.
￼
    # Arguments
        epochsize: Number of samples per epoch during training.
        batchsize: Number of samples per batch during training.
        start_epoch: First epoch where decay is applied.
        epochs_to_restart: Initial number of epochs before restarts.
        mult_factor: Increase of epochs_to_restart after each restart.
        lr_fac: Decrease of learning rate at epochs given in lr_reduction_epochs.
        lr_reduction_epochs: Fixed list of epochs at which to reduce learning rate.
￼
￼    # References
￼       - [SGDR: Stochastic Gradient Descent with Restarts](http://arxiv.org/abs/1608.03983)
￼    '''
    def __init__(self,
                epochsize,
                batchsize,
                epochs_to_restart=20,
                mult_factor=2,
                lr_fac=0.1,
                lr_reduction_epochs=(60, 120, 160)):
        super(SGDRScheduler, self).__init__()
        self.epoch = -1
        self.batch_since_restart = 0
        self.next_restart = epochs_to_restart
        self.epochsize = epochsize
        self.batchsize = batchsize
        self.epochs_to_restart = epochs_to_restart
        self.mult_factor = mult_factor
        self.batches_per_epoch = self.epochsize / self.batchsize
        self.lr_fac = lr_fac
        self.lr_reduction_epochs = lr_reduction_epochs
        self.lr_log = []

    def on_train_begin(self, logs={}):
        self.lr = K.get_value(self.model.optimizer.lr)
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1

    def on_batch_end(self, batch, logs={}):
        fraction_to_restart = self.batch_since_restart / \
            (self.batches_per_epoch * self.epochs_to_restart)
        lr = 0.5 * self.lr * (1 + np.cos(fraction_to_restart * np.pi))
        K.set_value(self.model.optimizer.lr, lr)
        logs['lr'] = lr

        self.batch_since_restart += 1
        self.lr_log.append(lr)
        
    def on_epoch_end(self, epoch, logs={}):
        if self.epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.epochs_to_restart *= self.mult_factor
            self.next_restart += self.epochs_to_restart

        if (self.epoch + 1) in self.lr_reduction_epochs:
            self.lr *= self.lr_fac


def get_callbacks(model_direction, epochsize, batchsize):
    '''
    返回 keras 的 callback list
    '''
    model_direction = Path(model_direction)
    csv_log_file = str(model_direction.parent / 'log' / 'model_train_log.csv')
    tensorboard_log_direction = str(model_direction.parent / 'log')
    ckpt_file = str(model_direction / 'ckpt_model.val_precision.{epoch:02d}-{val_precision:.4f}.h5')
    ckpt_file2 = str(model_direction / 'ckpt_model.val_f1.{epoch:02d}-{val_f1:.4f}.h5')

    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=200, verbose=0, mode='max')
    csv_log = CSVLogger(csv_log_file)
    checkpoint = ModelCheckpoint(ckpt_file, monitor='val_precision', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(ckpt_file2, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_direction, batch_size=batchsize)

    callbacks_list = [
        MetricCallback(predict_batch_size=batchsize),
        csv_log,
        early_stopping,
        checkpoint,
        checkpoint2,
        tensorboard_callback]
    return callbacks_list
