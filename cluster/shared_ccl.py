from keras.layers import Reshape, Lambda
from clustering_layer import tanh
from clustering_layer import Clustering, Clustering_centers
import tensorflow as tf
from keras.models import Sequential

def shared_ccl(input_dim = 512, cluster_num=10, cluster_dim=16, iter=3, shared_across_modal=False, modal=None):

    if shared_across_modal:
        seq = Sequential()
        seq.add(Reshape((-1, input_dim), input_shape=(None, None, input_dim)))
        seq.add(Lambda(tanh))
        seq.add(Clustering(cluster_num, cluster_dim, iter, True))
        seq.add(Lambda(tanh))

    else:
        seq = Sequential()
        input_shape = None
        if modal == 'image':
            input_shape = (8, 8, input_dim)
        elif modal == 'sound':
            input_shape = (31, 4, input_dim)
        seq.add(Reshape((-1, input_dim), input_shape=input_shape))
        seq.add(Lambda(tanh))
        seq.add(Clustering(cluster_num, cluster_dim, iter, False))
        seq.add(Lambda(tanh))

    return seq


def shared_ccl_centers(input_dim = 512, cluster_num=10, cluster_dim=16, iter=3, shared_across_modal=False, modal=None):

    if shared_across_modal:
        seq = Sequential()
        seq.add(Reshape((-1, input_dim), input_shape=(None, None, input_dim)))
        seq.add(Lambda(tanh))
        seq.add(Clustering_centers(cluster_num, cluster_dim, iter, True))

    else:
        seq = Sequential()
        input_shape = None
        if modal == 'image':
            input_shape = (8, 8, input_dim)
        elif modal == 'sound':
            input_shape = (31, 4, input_dim)
        seq.add(Reshape((-1, input_dim), input_shape=input_shape))
        seq.add(Lambda(tanh))
        seq.add(Clustering_centers(cluster_num, cluster_dim, iter, False))

    return seq