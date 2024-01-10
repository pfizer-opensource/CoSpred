import numpy as np
import tensorflow as tf
import keras.backend as k

class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, name="custom_metric", **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass
        # Use TensorFlow operations to update the state of the metric
        # based on the predictions and the true labels

    def result(self):
        return self.total

    def reset_state(self):
        self.total.assign(0.)


def spectral_distance(pred, true):
    pred_norm = k.l2_normalize(pred, axis=-1)
    true_norm = k.l2_normalize(true, axis=-1)
    product = k.sum(pred_norm * true_norm, axis=1)
    arccos = tf.acos(product)
    return 2 * arccos / np.pi


def masked_spectral_distance(pred, true):
    epsilon = k.epsilon()
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = k.l2_normalize(pred_masked, axis=-1)
    true_norm = k.l2_normalize(true_masked, axis=-1)
    product = k.sum(pred_norm * true_norm, axis=1)
    arccos = tf.acos(product)
    return 2 * arccos / np.pi


def pearson_corr(pred, true):
    pred_norm = k.l2_normalize(pred, axis=-1)
    true_norm = k.l2_normalize(true, axis=-1)
    product = k.sum(pred_norm * true_norm, axis=1)
    corr = product / tf.sqrt(tf.reduce_sum(pred_norm ** 2) * tf.reduce_sum(true_norm ** 2))
    return corr


def cos_sim(pred, true):
    pred_norm = tf.nn.l2_normalize(pred, axis=-1)
    true_norm = tf.nn.l2_normalize(true, axis=-1)
    cos_sim = tf.reduce_sum(pred_norm * true_norm, axis=-1)
    return cos_sim


def binarize(y, threshold):
    """
    removes peaks less than a threshold
    """
    y_binary = tf.identity(y)
    y_binary = tf.cast(y_binary >= threshold, tf.float32)
    # y_binary = y.clone()
    # y_binary[y_binary < threshold] = 0.0
    # y_binary[y_binary >= threshold] = 1.0
    return y_binary


class ComputeMetrics(CustomMetric):
    def __init__(self, name="compute_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.binarize_threshold = 0.02

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use TensorFlow operations to update the state of the metric
        # based on the predictions and the true labels
        # true_binary = tf.cast(y_true > self.binarize_threshold, tf.float32)
        # pred_binary = tf.cast(y_pred > self.binarize_threshold, tf.float32)
        true_binary = binarize(y_true, threshold=self.binarize_threshold)
        pred_binary = binarize(y_pred, threshold=self.binarize_threshold)

        # mass prediction focused metrics
        self._tp = tf.reduce_sum(tf.cast((pred_binary == 1.0) & (true_binary == 1.0), tf.float32), axis=1)  # True positive
        self._tn = tf.reduce_sum(tf.cast((pred_binary == 0.0) & (true_binary == 0.0), tf.float32), axis=1)  # True negative
        self._fp = tf.reduce_sum(tf.cast((pred_binary == 1.0) & (true_binary == 0.0), tf.float32), axis=1)  # False positive
        self._fn = tf.reduce_sum(tf.cast((pred_binary == 0.0) & (true_binary == 1.0), tf.float32), axis=1)  # False negative

        self.accuracy = (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
        self.sensitivity = self._tp / (self._tp + self._fn)
        self.specificity = self._tn / (self._tn + self._fp)
        self.balanced_accuracy = (self.sensitivity + self.specificity) / 2.0
        self.precision = self._tp / (self._tp + self._fp)
        self.recall = self._tp / (self._tp + self._fn)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        
        # intensity focused metrics
        self.spectral_distance = spectral_distance(y_pred, y_true)
        self.masked_spectral_distance = masked_spectral_distance(y_pred, y_true)
        self.pcc= pearson_corr(y_pred, y_true)
        self.cos_sim = cos_sim(y_pred, y_true)

        batch_size = tf.shape(pred_binary)[0]

        # top1_mass = tf.argsort(y_true, direction='DESCENDING')[:, -1]
        # top2_mass = tf.argsort(y_true, direction='DESCENDING')[:, -2]
        # top3_mass = tf.argsort(y_true, direction='DESCENDING')[:, -3]
        # top4_mass = tf.argsort(y_true, direction='DESCENDING')[:, -4]
        # top5_mass = tf.argsort(y_true, direction='DESCENDING')[:, -5]
        top1_mass = tf.argsort(y_true)[:, -1]
        top2_mass = tf.argsort(y_true)[:, -2]
        top3_mass = tf.argsort(y_true)[:, -3]
        top4_mass = tf.argsort(y_true)[:, -4]
        top5_mass = tf.argsort(y_true)[:, -5]

        self.top1_out_1 = tf.gather(pred_binary, top1_mass, batch_dims=1)

        count_peaks = tf.gather(pred_binary, top1_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top2_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top3_mass, batch_dims=1)

        self.top1_out_3 = tf.cast(count_peaks >= 1.0, tf.float32)
        self.top2_out_3 = tf.cast(count_peaks >= 2.0, tf.float32)
        self.top3_out_3 = tf.cast(count_peaks >= 3.0, tf.float32)

        count_peaks = tf.gather(pred_binary, top1_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top2_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top3_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top4_mass, batch_dims=1) + \
                    tf.gather(pred_binary, top5_mass, batch_dims=1)

        self.top1_out_5 = tf.cast(count_peaks >= 1.0, tf.float32)
        self.top2_out_5 = tf.cast(count_peaks >= 2.0, tf.float32)
        self.top3_out_5 = tf.cast(count_peaks >= 3.0, tf.float32)
        self.top4_out_5 = tf.cast(count_peaks >= 4.0, tf.float32)
        self.top5_out_5 = tf.cast(count_peaks >= 5.0, tf.float32)

    def return_metrics(self):
        metrics_dict = {'balanced_accuracy': self.balanced_accuracy,
                        'true_positive': self._tp,
                        'true_negative': self._tn,
                        'false_positive': self._fp,
                        'false_negative': self._fn,
                        'precision': self.precision,
                        'recall': self.recall,
                        'sensitivity': self.sensitivity,
                        'specificity': self.specificity,
                        'F1': self.F1,
                        'top1_out_1': self.top1_out_1,
                        'top1_out_3': self.top1_out_3,
                        'top2_out_3': self.top2_out_3,
                        'top3_out_3': self.top3_out_3,
                        'top1_out_5': self.top1_out_5,
                        'top2_out_5': self.top2_out_5,
                        'top3_out_5': self.top3_out_5,
                        'top4_out_5': self.top4_out_5,
                        'top5_out_5': self.top5_out_5,
                        'spectral_distance': self.spectral_distance,
                        'masked_spectral_distance': self.masked_spectral_distance,
                        'pearson_correlation': self.pcc,
                        'cosine_similarity': self.cos_sim
                        }
        return metrics_dict

    def result(self):
        metrics = self.return_metrics()
        for key, value in metrics.items():
            metrics[key] = tf.reduce_mean(value)
        return metrics

    # def reset_state(self):
    #     self.total.assign(0.)





