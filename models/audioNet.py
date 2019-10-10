from keras.layers import Input, Conv2D, MaxPooling2D,AveragePooling2D, Dense, Flatten, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout
from keras.models import Model


def audio_subnet_vggish_dmc(input_shape, load_weights=False):
    audio_input = Input(shape=input_shape, name='audio_input')

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block1_conv1')(audio_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='audio_block1_pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='audio_block2_pool1')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block3_conv1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block3_conv2')(x)
    x = MaxPooling2D((2, 2),  strides=(2, 2), padding='same', name='audio_block3_pool1')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block5_conv1')(x)
    audio_output = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='audio_block5_conv2')(x)

    #audio_output = GlobalMaxPooling2D()(x)

    #audio_output = Dense(512, activation='relu', name='audio_fc')(x)

    #######################################################
    #x = Flatten(name='audio_flatten')(x)
    #x = Dense(4096, activation='relu', name='audio_fc1')(x)
    #audio_output = Dense(4096, activation='relu', name='audio_fc2')(x)

    #######################################################

    model = Model(audio_input, audio_output, name='audio_subnet')

    if load_weights:
        model.load_weights('model_temp/vggish_pure_conv.h5')

    return model