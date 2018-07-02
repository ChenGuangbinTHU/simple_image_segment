
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight 



from keras.models import *
from keras.layers import *
import keras.layers.merge
import tensorflow as tf
import sys
sys.path.insert(1, './src')
# from crfrnn_layer import CrfRnnLayer
from keras.backend import permute_dimensions
from keras import backend as K
from models.resnet_code import resnet_keras



import os
file_path = os.path.dirname( os.path.abspath(__file__) )

VGG_Weights_path = file_path+"/data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_first' 


def resnet_aspp( nClasses ,  input_height=416, input_width=608 , vgg_level=3):

	# assert input_height%32 == 0
	# assert input_width%32 == 0

	img_input = Input(shape=(3,input_height,input_width))
	p5 = resnet_keras.resnet_18_output(img_input, [2, 2, 2,2])
	# branching for Atrous Spatial Pyramid Pooling
	# hole = 6
	# b1 = ZeroPadding2D(padding=(6, 6))(p5)
	n = 36
	b1 = Conv2D(10, (3, 3), padding='same',dilation_rate=(n, n), activation='relu', name='fc6_1', data_format=IMAGE_ORDERING)(p5)
	b1 = Dropout(0.5)(b1)
	# b1 = Conv2D(10, (1, 1), activation='relu', name='fc7_1', data_format=IMAGE_ORDERING)(b1)
	# b1 = Dropout(0.5)(b1)
	# b1 = Conv2D(2, (1, 1), activation='relu', name='fc8_voc12_1', data_format=IMAGE_ORDERING)(b1)

	# hole = 12
	# b2 = ZeroPadding2D(padding=(12, 12))(p5)
	b2 = Conv2D(10, (3, 3), padding='same',dilation_rate=(n*2, n*2), activation='relu', name='fc6_2', data_format=IMAGE_ORDERING)(p5)
	b2 = Dropout(0.5)(b2)
	# b2 = Conv2D(10, (1, 1), activation='relu', name='fc7_2', data_format=IMAGE_ORDERING)(b2)
	# b2 = Dropout(0.5)(b2)
	# b2 = Conv2D(2, (1, 1), activation='relu', name='fc8_voc12_2', data_format=IMAGE_ORDERING)(b2)

	# hole = 18
	# b3 = ZeroPadding2D(padding=(18, 18))(p5)
	b3 = Conv2D(10, (3, 3), padding='same',dilation_rate=(n*3, n*3), activation='relu', name='fc6_3', data_format=IMAGE_ORDERING)(p5)
	b3 = Dropout(0.5)(b3)
	# b3 = Conv2D(10, (1, 1), activation='relu', name='fc7_3', data_format=IMAGE_ORDERING)(b3)
	# b3 = Dropout(0.5)(b3)
	# b3 = Conv2D(2, (1, 1), activation='relu', name='fc8_voc12_3', data_format=IMAGE_ORDERING)(b3)

	# hole = 24
	# b4 = ZeroPadding2D(padding=(24, 24))(p5)
	b4 = Conv2D(10, (3, 3), padding='same',dilation_rate=(n*4, n*4), activation='relu', name='fc6_4', data_format=IMAGE_ORDERING)(p5)
	b4 = Dropout(0.5)(b4)
	# b4 = Conv2D(10, (1, 1), activation='relu', name='fc7_4', data_format=IMAGE_ORDERING)(b4)
	# b4 = Dropout(0.5)(b4)
	# b4 = Conv2D(2, (1, 1), activation='relu', name='fc8_voc12_4', data_format=IMAGE_ORDERING)(b4)
	
	
	logits = add([b1, b2, b3, b4])#remove
	# out = UpSampling2D(size=(8,8), data_format=IMAGE_ORDERING)(logits)
	
	# logits = BilinearUpsampling(upsampling=upsampling)(s)
	'''
	def mul_minus_one(a):
		a = Permute((2, 3, 1))(a)
		a = K.tf.image.resize_bilinear(a,(input_height, input_width),align_corners=True)
		a = Permute((3, 1, 2))(a)
		return a
	def mul_minus_one_output_shape(input_shape):
		return input_shape
	'''
	def mul_minus_one(a):
		return K.resize_images(a,8, 8, data_format=IMAGE_ORDERING)
	def mul_minus_one_output_shape(input_shape):
		return input_shape
	

	# out = (Activation('softmax'))(logits)
	resize = Lambda(mul_minus_one)
	# out = resize(logits)
	out = Conv2DTranspose(2 , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING ,padding='same')(logits)
	
	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	inputs = img_input
	
	# Create model.
	# model = Model(inputs, out, name='deeplabV2')
	
	o_shape = Model(img_input ,  out).output_shape
	# print(o_shape)
	# exit(0)
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]
	# print(o_shape)
	
	o = (Reshape((  -1  , outputHeight*outputWidth   )))(out)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	model.summary()
	# exit(0)
	exp_model=model
	return model




if __name__ == '__main__':
	m = deeplabv2_resnet( 101 )
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')
