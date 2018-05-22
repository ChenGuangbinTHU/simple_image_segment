from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
from keras.layers import *
import numpy as np

IMAGE_ORDERING = 'channels_first' 

def start_block(inputs):
    outputs = Conv2D(64, (7, 7), strides = (2,2), padding='same', data_format=IMAGE_ORDERING)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')( outputs)
    outputs = MaxPooling2D((3, 3), (2, 2), padding = 'same', data_format=IMAGE_ORDERING)(outputs)
    # outputs = Activation('softmax')(outputs)
    return outputs

def basic_block(inputs, num_o, half_size = False, identity_connection = True):
    first_s = 2 if half_size else 1
    assert num_o % 4 == 0, 'number of output ERROR'

    #branch 1
    if not identity_connection:
        o_b1 = Conv2D(num_o, (1, 1), strides=(first_s, first_s), data_format=IMAGE_ORDERING)(inputs)
    else:
        o_b1 = inputs

    #branch2
    o_b2 = Conv2D(num_o, (3, 3), strides=(first_s, first_s), padding='same', data_format=IMAGE_ORDERING)(inputs)
    o_b2 = BatchNormalization()(o_b2)
    o_b2 = Conv2D(num_o, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(o_b2)
    o_b2 = BatchNormalization()(o_b2)
    output = add([o_b1, o_b2])
    output = Activation('relu')(output)
    return output

def dilated_basic_block(inputs, num_o, dilated_rate, half_size = False, identity_connection = True):
    first_s = 2 if half_size else 1
    assert num_o % 4 == 0, 'number of output ERROR'

    #branch 1
    if not identity_connection:
        o_b1 = Conv2D(num_o, (1, 1), strides=(first_s, first_s), data_format=IMAGE_ORDERING)(inputs)
    else:
        o_b1 = inputs

    #branch2
    o_b2 = Conv2D(num_o, (3, 3), strides=(first_s, first_s), padding='same', data_format=IMAGE_ORDERING)(inputs)
    o_b2 = BatchNormalization()(o_b2)
    o_b2 = Conv2D(num_o, (3, 3), dilation_rate=(dilated_rate, dilated_rate), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(o_b2)
    o_b2 = BatchNormalization()(o_b2)
    output = add([o_b1, o_b2])
    output = Activation('relu')(output)
    return output

def resnet_18_output(inputs):
    o = start_block(inputs)
    for i in range(0, 4):
        for j in range(2):
            if i < 2:
                o = basic_block(o, 16*2**i, i == 1 and j == 0,not j == 0)
            else:
                o = dilated_basic_block(o, 16*2**i, 2, i == 1 and j == 0,not j == 0)
    return o

if __name__ == '__main__':
    img_input = Input(shape=(512,512,3))
    a = resnet_18_output(img_input)
    # a = dilated_basic_block(a, 64, 2)
    m = Model(img_input, a)
    m.summary()