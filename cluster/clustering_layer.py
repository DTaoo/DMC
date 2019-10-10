#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.layers.merge import dot


def tanh(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.tanh(10 * K.sqrt(s_squared_norm)) / K.sqrt(s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


class Clustering(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, **kwargs):
        super(Clustering, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

    def build(self, input_shape):
        super(Clustering, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) 
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = K.sum(o, axis=1)   # the previous version does not have this op, due to the recent updates of keras
            if i < self.routings - 1:
                o = K.l2_normalize(o,-1)  # here is for normalize  if it not works, remove this line
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = K.sum(b, axis=1)  # the previous version does not have this op, due to the recent updates of keras

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule':self.dim_capsule,
            'routings': self.routings,
            'share_weights':self.share_weights
        }
        base_config = super(Clustering, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Clustering_centers(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, **kwargs):
        super(Clustering_centers, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

    def build(self, input_shape):
        super(Clustering_centers, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])


        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = K.sum(o, axis=1)   # the previous version does not have this op, due to the recent updates of keras
            if i < self.routings - 1:
                # o = K.l2_normalize(o,-1)  # here is for normalize  if it does not work, remove this line
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = K.sum(b, axis=1)  # the previous version does not have this op, due to the recent updates of keras

        x = tf.concat([o,c],2)
        return x

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, 496+16)


    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule':self.dim_capsule,
            'routings': self.routings,
            'share_weights':self.share_weights
        }
        base_config = super(Clustering_centers, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))