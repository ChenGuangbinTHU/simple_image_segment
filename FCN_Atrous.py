
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight 



from keras.models import *
from keras.layers import *
import tensorflow as tf
import sys
sys.path.insert(1, './src')
from crfrnn_layer import CrfRnnLayer
from keras.backend import permute_dimensions


import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_first' 


def FCN8_Atrous( nClasses ,  input_height=416, input_width=608 , vgg_level=3):

	# assert input_height%32 == 0
	# assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(3,input_height,input_width))

	x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	x_exp = Flatten(name='f')(x)
	x_exp = Dense(2)(x_exp)
	x_exp = (Activation('softmax'))(x_exp)
	# Block 2
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	h = MaxPooling2D((3, 3), strides=(1, 1), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	h = ZeroPadding2D(padding=(2, 2))(h)
	h = AtrousConvolution2D(512, (3, 3), atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
	h = ZeroPadding2D(padding=(2, 2))(h)
	h = AtrousConvolution2D(512, (3, 3), atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
	h = ZeroPadding2D(padding=(2, 2))(h)
	h = AtrousConvolution2D(512, (3, 3), atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
	h = ZeroPadding2D(padding=(1, 1))(h)
	p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

	# branching for Atrous Spatial Pyramid Pooling
	# hole = 6
	b1 = ZeroPadding2D(padding=(6, 6))(p5)
	b1 = AtrousConvolution2D(1024, (3, 3), atrous_rate=(6, 6), activation='relu', name='fc6_1')(b1)
	b1 = Dropout(0.5)(b1)
	b1 = Convolution2D(1024, (1, 1), activation='relu', name='fc7_1')(b1)
	b1 = Dropout(0.5)(b1)
	b1 = Convolution2D(2, (1, 1), activation='relu', name='fc8_voc12_1')(b1)

	# hole = 12
	b2 = ZeroPadding2D(padding=(12, 12))(p5)
	b2 = AtrousConvolution2D(1024, (3, 3), atrous_rate=(12, 12), activation='relu', name='fc6_2')(b2)
	b2 = Dropout(0.5)(b2)
	b2 = Convolution2D(1024, (1, 1), activation='relu', name='fc7_2')(b2)
	b2 = Dropout(0.5)(b2)
	b2 = Convolution2D(2, (1, 1), activation='relu', name='fc8_voc12_2')(b2)

	# hole = 18
	b3 = ZeroPadding2D(padding=(18, 18))(p5)
	b3 = AtrousConvolution2D(1024, (3, 3), atrous_rate=(18, 18), activation='relu', name='fc6_3')(b3)
	b3 = Dropout(0.5)(b3)
	b3 = Convolution2D(1024, (1, 1), activation='relu', name='fc7_3')(b3)
	b3 = Dropout(0.5)(b3)
	b3 = Convolution2D(2, (1, 1), activation='relu', name='fc8_voc12_3')(b3)

	# hole = 24
	b4 = ZeroPadding2D(padding=(24, 24))(p5)
	b4 = AtrousConvolution2D(1024, (3, 3), atrous_rate=(24, 24), activation='relu', name='fc6_4')(b4)
	b4 = Dropout(0.5)(b4)
	b4 = Convolution2D(1024, (1, 1), activation='relu', name='fc7_4')(b4)
	b4 = Dropout(0.5)(b4)
	b4 = Convolution2D(2, (1, 1), activation='relu', name='fc8_voc12_4')(b4)

	logits = merge([b1, b2, b3, b4], mode='sum')
	# logits = BilinearUpsampling(upsampling=upsampling)(s)


	out = (Activation('softmax'))(logits)


	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	inputs = img_input

	# Create model.
	model = Model(inputs, out, name='deeplabV2')

	model.summary()
	exit(0)
	return model, exp_model




if __name__ == '__main__':
	m = FCN8( 101 )
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')
