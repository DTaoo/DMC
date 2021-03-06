from keras import backend as T
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import dot



class Cosine(Layer):

    def __init__(self, output_dim=1,  **kwargs):
        self.output_dim = output_dim
        super(Cosine, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Cosine, self).build(input_shape)

    def call(self, x):

        o1 = o2 = tf.shape(x)[1] // 2

        H1 = x[:,0:o1]
        H2 = x[:,o1:o1 + o2]

        H1 = K.l2_normalize(H1, axis=-1)
        H2 = K.l2_normalize(H2, axis=-1)

        output = K.sum(H1 * H2, axis=-1)

        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
        }
        base_config = super(Cosine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
