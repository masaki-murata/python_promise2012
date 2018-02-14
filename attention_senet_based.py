from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda
from keras.layers.convolutional import Conv3D, Deconv3D, ZeroPadding3D, UpSampling3D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
import keras.backend as K
import numpy as np
import os



def attetion(x):

    feature_map = x[0]
    coef = x[1]
    coef = K.expand_dims(K.expand_dims(coef, axis=-2) ,axis=-2)

    x = coef*feature_map

    return x


def conv_block_unet(x, f, strides=(2,2,2)):

    x = LeakyReLU(0.2)(x)
    x = Conv3D(f, (3, 3, 3), strides=strides, padding="same")(x)
    x = BatchNormalization(axis=-1)(x)

    return x



def up_conv_block_seunet(x, x2, f, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(f, (3, 3, 3), padding="same")(x)
    x = BatchNormalization(axis=-1)(x)
    if dropout:
        x = Dropout(0.5)(x)

    channels_nb = K.int_shape(x2)[-1]

    channels_nb_bottleneck = channels_nb // 16

    x3=GlobalMaxPooling2D()(x2)
    x3 = Dense(channels_nb_bottleneck, activation='relu')(x3)
    x3 = Dense(channels_nb, activation='sigmoid')(x3)

    y = Lambda(lambda x: attetion(x))([x2, x3])

    x = Concatenate(axis=-1)([x, y])

    return x

def seunet(img_dims):

    filter_list = [64, 128, 256, 320, 320, 320, 320, 320]#[64 * min(5, (2 ** i)) for i in range(8)]

    unet_input = Input(shape=img_dims)

    encoder_layers = [Conv3D(filter_list[0], (3, 3, 3),
                           strides=(2, 2, 2), padding="same")(unet_input)]

    for i, f in enumerate(filter_list[1:]):
        conv = conv_block_unet(encoder_layers[-1], f)
        encoder_layers.append(conv)




    filter_list = [320, 320, 320, 256, 128, 64, 64]



    decoder_layers = [up_conv_block_seunet(encoder_layers[-1], encoder_layers[-2],
                                       filter_list[0], dropout=True)]

    for i, f in enumerate(filter_list[1:]):
        if i < 2:
            d_o = True
        else:
            d_o = False
        conv = up_conv_block_seunet(decoder_layers[-1], encoder_layers[-(i + 3)], f, dropout=d_o)
        decoder_layers.append(conv)

    x = Activation("relu")(decoder_layers[-1])
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(4, (3, 3, 3), padding="same")(x)
    x = Activation("sigmoid")(x)

    unet_output = Model(inputs=[unet_input], outputs=[x])

    unet_output.summary()

    return unet_output
