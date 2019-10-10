## This is a primary version of DMC, a completed version will be released after Nov.2019. ##

### head start ###
import sys
sys.path.append('..')

# sci relevant
import numpy as np
import scipy.io as sio
import linecache

from videoNet import image_subnet_dmc
from audioNet import audio_subnet_vggish_dmc

# from utils import load_training_data
from keras.layers import GlobalAveragePooling2D, Dense, concatenate, Reshape, Activation, Input, Lambda, GlobalMaxPooling2D, Softmax, multiply, dot
### head complete ###
from keras.models import Model
from keras.optimizers import adam, SGD
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from cluster.shared_ccl import shared_ccl
from cluster.cosine_layer import Cosine
from utils.multimodal_loader import sound_image_generator_cosine
import random
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.models import load_model


def constant_loss(y_true, y_pred):
    return y_pred

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)



if __name__ == '__main__':
    ######################## Parameters Setting Start ##############################

    # the size of the new space, input to the CCA analysis
    outdim_size = 10

    # size of the input for image and sound
    image_input_size = (256, 256, 3)
    sound_input_size = (None, None, 1)

    # if include the top FC layer for both VGG-16 and VGGish
    #  VGG-16 for image and VGGish for sound
    include_top = False

    # the parameters for training the network
    learning_rate = 1e-3
    nb_epoch = 30
    mini_epoch = 10  # 100

    # the sampled number of samples for batch generator
    batch_size = 1000

    # the number of samples within each mini-batch after batch generator
    minibatch_size = 10  # 1000

    # total samples
    sample_total = 2000000

    # pretrain or not
    pretrain = False

    # clustering
    cluster_num = 10
    cluster_dim = 16
    cluster_iter = 3
    ######################## Parameters Setting End ##############################


    real_sound = Input(shape=sound_input_size)
    fake_sound = Input(shape=sound_input_size)

    vggish_model = audio_subnet_vggish_dmc(input_shape=sound_input_size)
    real_sound_embedding = vggish_model(real_sound)
    fake_sound_embedding = vggish_model(fake_sound)


    shared_ccl_layer = shared_ccl(input_dim=512,
                                 cluster_num=cluster_num,
                                 cluster_dim=cluster_dim,
                                 iter=cluster_iter,
                                 shared_across_modal=True)
    real_sound_cluster = shared_ccl_layer(real_sound_embedding)
    fake_sound_cluster = shared_ccl_layer(fake_sound_embedding)

    #with tf.device('/gpu:1'):
    # create vgg model and obtain the image_output
    real_image = Input(shape=image_input_size)


    # it is better to use pretrained vggnet as the start point, as well as verifying the model
    vgg_model = image_subnet_dmc(input_shape=image_input_size)
    real_image_embedding = vgg_model(real_image)


    real_image_cluster = shared_ccl_layer(real_image_embedding)

    real_cluster_layer = concatenate([real_sound_cluster, real_image_cluster], name='real_cluster_layer')
    fake_cluster_layer = concatenate([fake_sound_cluster, real_image_cluster], name='fake_cluster_layer')
    right_cos = Cosine()(real_cluster_layer)
    wrong_cos = Cosine()(fake_cluster_layer)


    #right_cos = K.sum(dot([real_sound_cluster, real_image_cluster], -1, normalize=True), -1)
    #wrong_cos = K.sum(dot([fake_sound_cluster, real_image_cluster], -1, normalize=True), -1)


    loss_cluster = Lambda(lambda x: K.relu(cluster_num + x[0] - x[1]))(
        [wrong_cos, right_cos])


    dmc_model = Model(inputs=[real_sound, real_image, fake_sound], outputs=[loss_cluster])

    #dmc_model_name = '/mount/hudi/moe/dmc/model_saved/shared_v_to_a_finetune/' \
    #                 'shared-v-to-a-finetune-5-folder-0-cluser-10-dim-16-route-3-01.hdf5'
    #dmc_model.load_weights(dmc_model_name)


    adam_ = adam(lr=1e-4)
    sgd = SGD(momentum=0.9)
    #parallel_model = multi_gpu_model(dmc_model, gpus=3)
    dmc_model.compile(optimizer=adam_,
                      loss=[identity_loss])


    ######################## Network training Start ###############################
    # load files
    data_file = 'train_videos.txt'
    lines = linecache.getlines(data_file)

    # randomly select samples from the data file
    # lines = lines[:400000]
    random.seed(18)
    lines = random.sample(lines, 400000)  # more examples

    folder_num = 40
    batch = len(lines) / folder_num
    for k in range(100):
        j = k
        print "current iteration %d " % j
        for i in range(folder_num):
            print "current processing %d forder" % i
            current_lines = lines[i * batch:(i + 1) * batch]

            current_num = "shared-v-to-a-finetune-%d-folder-%d-cluser-%d-dim-%d-route-%d-" % (j, i, cluster_num, cluster_dim, cluster_iter)
            filepath = "model_saved/shared_v_to_a_finetune/" + "" + current_num + "{epoch:02d}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='mean_pred', verbose=1)

            hist = dmc_model.fit_generator(sound_image_generator_cosine(current_lines, batch / 1000),
                                            steps_per_epoch=1000,
                                            epochs=1,
                                            shuffle=False,
                                            callbacks=[checkpoint],
                                            max_queue_size=30
                                           )

    ######################## Network training End ###############################
