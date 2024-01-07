# -*- coding: utf-8 -*-
"""Variational autoencoder test script

Author: Stephen Ra (stephen.ra@pfizer.com)
"""

import argparse
import numpy as np

from keras import callbacks
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import losses
from keras.datasets import mnist
from keras.callbacks import TensorBoard

BATCH_SIZE = 200
ORIGINAL_DIM = 784
LATENT_DIM = 2
INTERMEDIATE_DIM = 256
EPOCHS = 50
EPSILON_STD = 1.0

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

x = Input(batch_shape=(BATCH_SIZE, ORIGINAL_DIM))
h = Dense(INTERMEDIATE_DIM, activation='relu')(x)
z_mean = Dense(LATENT_DIM)(h)
z_log_sigma = Dense(LATENT_DIM)(h)


def get_arguments():
    """Model arguments"""
    parser = argparse.ArgumentParser(description='Antibody Property Model')
    parser.add_argument('output_name', type=str, help='src/log/')
    return parser.parse_args()


def sampling(args):
    """Sampling z from a unit Gaussian"""
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(
        shape=(BATCH_SIZE, LATENT_DIM), mean=0., stddev=EPSILON_STD)
    return z_mean + K.exp(z_log_sigma) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(LATENT_DIM, ))([z_mean, z_log_sigma])


def main():
    """
    Fits VAE
    """
    decoder_h = Dense(INTERMEDIATE_DIM, activation='relu')
    decoder_mean = Dense(ORIGINAL_DIM, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        '''
        Simple binary cross-entropy loss function
        KL-divergence is defined according to Kingma and Welling, 2014
        by taking mean over the latent dimensions
        '''
        xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.mean(
            1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    # callback for loss reduction
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=2, min_lr=0, verbose=1)

    args = get_arguments()

    vae.fit(
        x_train,
        x_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr, TensorBoard(log_dir=args.output_name)],
        validation_data=(x_test, x_test))


if __name__ == '__main__':
    main()
