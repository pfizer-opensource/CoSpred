from __future__ import print_function
import tensorflow as tf
print("Tensorflow version: {}".format(tf.__version__))
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
