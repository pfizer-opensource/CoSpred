import numpy as np
import tensorflow as tf
import keras.backend as k

def masked_spectral_distance(pred, true):
    # Note, fragment ions that cannot exists (i.e. y20 for a 7mer) must have the value  -1.
    epsilon = k.epsilon()
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    pred_norm = k.l2_normalize(pred_masked, axis=-1)
    true_norm = k.l2_normalize(true_masked, axis=-1)
    # make sure the tensors are in float16 to save memory
    pred_norm = tf.cast(pred_norm, tf.float16)
    true_norm = tf.cast(true_norm, tf.float16)
    product = k.sum(pred_norm * true_norm, axis=1)
    arccos = tf.acos(product)
    return 2 * arccos / np.pi


def spectral_distance(pred, true):
    pred_norm = k.l2_normalize(pred, axis=-1)
    true_norm = k.l2_normalize(true, axis=-1)
    # make sure the tensors are in float16 to save memory
    pred_norm = tf.cast(pred_norm, tf.float16)
    true_norm = tf.cast(true_norm, tf.float16)
    product = k.sum(pred_norm * true_norm, axis=1)
    arccos = tf.acos(product)
    return 2 * arccos / np.pi


# losses = {"masked_spectral_distance": masked_spectral_distance}
losses = {"spectral_distance": spectral_distance}


def get(loss_name):
    if loss_name in losses:
        return losses[loss_name]
    else:
        return loss_name
