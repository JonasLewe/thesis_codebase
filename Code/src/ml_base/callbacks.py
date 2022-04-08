import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, jaccard_score, f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

class CalcAndLogF1Score(tf.keras.callbacks.Callback):
    def __init__(self, valid_data_generator):
        super(CalcAndLogF1Score, self).__init__()
        self.validation_data = valid_data_generator

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_steps_per_epoch = np.math.ceil(self.validation_data.samples / self.validation_data.batch_size)
        
        predict = self.model.predict(self.validation_data, steps=val_steps_per_epoch)
        val_predict = [1 * (x[0]>=0.5) for x in predict]
        
        val_targ = self.validation_data.classes

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(f" — val_f1: {_val_f1:.4f} — val_precision: {_val_precision:.4f} — val_recall: {_val_recall:.4f}")
        

class JaccardScoreCallback(tf.keras.callbacks.Callback):
    """Computes the Jaccard score and logs the results to TensorBoard."""

    def __init__(self, model, valid_data_generator, log_dir):
        self.model = model
        self.validation_data = valid_data_generator
        self.keras_metric = tf.keras.metrics.Mean("jaccard_score")
        self.epoch = 0
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, model.name))

    def on_epoch_end(self, batch, logs=None):
        self.epoch += 1
        self.keras_metric.reset_states()
        
        val_steps_per_epoch = np.math.ceil(self.validation_data.samples/self.validation_data.batch_size)
        
        predict = self.model.predict(self.validation_data, steps=val_steps_per_epoch)
        val_predict = [1 * (x[0]>=0.5) for x in predict]
        
        val_targ = self.validation_data.classes
        
        jaccard_value = jaccard_score(val_predict, val_targ, average=None)
        self.keras_metric.update_state(jaccard_value)
        self._write_metric(self.keras_metric.name, self.keras_metric.result().numpy().astype(float))

    def _write_metric(self, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=self.epoch,)
            self.summary_writer.flush()

class F1_Metric(tf.keras.callbacks.Callback):
    def __init__(self, model, valid_data_generator, filepath='./saved_models'):
        self.model = model
        self.validation_data = valid_data_generator
        self.file_path = filepath
                

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        
        # check if model already exists
        for fname in os.listdir(self.file_path):
            if fname.endswith('.h5'):
                self.best_val_f1 = float(fname.split('_')[-2])
                print(f"Previous f1_score: {self.best_val_f1}")
            else:
                self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        val_steps_per_epoch = np.math.ceil(self.validation_data.samples/self.validation_data.batch_size)
        predict = self.model.predict(self.validation_data, steps=val_steps_per_epoch)
        val_predict = [1 * (x[0]>=0.5) for x in predict]
        val_targ = self.validation_data.classes
        
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f"_val_f1: {_val_f1:.4f}, _val_precision: {_val_precision:.4f}, _val_recall: {_val_recall:.4f}")
        print(f"max f1: {max(self.val_f1s):.4f}")
        if _val_f1 > self.best_val_f1:
            
            # check if model already exists
            for fname in os.listdir(self.file_path):
                if fname.endswith('.h5'):
                    if (_val_f1 > float(fname.split('_')[-2])):
                        os.remove(f"{self.file_path}/{fname}")
                        self.model.save(f"{self.file_path}/base_model_{_val_f1:.4f}_.h5", overwrite=True)
            
            self.best_val_f1 = _val_f1
            print(f"best f1: {self.best_val_f1:.4f}")
        else:
            print(f"val f1: {_val_f1:.4f}, but not the best f1")
        return

