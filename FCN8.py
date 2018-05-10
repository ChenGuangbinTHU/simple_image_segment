
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

# crop o1 wrt o2
def crop( o1 , o2 , i  ):
	o_shape2 = Model( i  , o2 ).output_shape
	outputHeight2 = o_shape2[2]
	outputWidth2 = o_shape2[3]

	o_shape1 = Model( i  , o1 ).output_shape
	outputHeight1 = o_shape1[2]
	outputWidth1 = o_shape1[3]

	cx = abs( outputWidth1 - outputWidth2 )
	cy = abs( outputHeight2 - outputHeight1 )

	if outputWidth1 > outputWidth2:
		o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
	else:
		o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
	
	if outputHeight1 > outputHeight2 :
		o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
	else:
		o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

	return o1 , o2 

def FCN8( nClasses ,  input_height=416, input_width=608 , vgg_level=3):

	# assert input_height%32 == 0
	# assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(3,input_height,input_width))

	x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	x_exp = Flatten(name='f')(x)
	x_exp = Dense(2)(x_exp)
	x_exp = (Activation('softmax'))(x_exp)
	# Block 2
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dense(1024, activation='relu', name='fc2')(x)
	
	exp_model = Model(img_input, x_exp)
	x = Dense( 250 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	# vgg.load_weights(VGG_Weights_path)

	o = f5

	o = ( Conv2D( 512 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
	o = Dropout(0.5)(o)
	o = ( Conv2D( 512 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
	o = Dropout(0.5)(o)

	o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
	o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)

	o2 = f4
	o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
	
	o , o2 = crop( o , o2 , img_input )
	
	o = Add()([ o , o2 ])

	o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
	o2 = f3 
	o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
	o2 , o = crop( o2 , o , img_input )
	o  = Add()([ o2 , o ])


	o = Conv2DTranspose( nClasses , kernel_size=(16,16) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)
	print(o.shape)
	# print(o.shape)
	upscore = Cropping2D(((4, 4), (4, 4)))(Permute((2,3,1))(o))
	# print(upscore)
	# print(tf.transpose(o,(0,2,3,1)).shape)
	# print(img_input.shape)
	# print(upscore.shape)
	# exit(0)
	# o = CrfRnnLayer(image_dims=(input_height, input_width),
    #                      num_classes=nClasses,
    #                      theta_alpha=160.,
    #                      theta_beta=3.,
    #                      theta_gamma=3.,
    #                      num_iterations=10,
    #                      name='crfrnn')([upscore, Permute((2,3,1))(img_input)])
	# o = Permute((3,1,2))(o)

	o_shape = Model(img_input ,  o).output_shape
	# print(o_shape)
	# exit(0)
	outputHeight = o_shape[2]
	outputWidth = o_shape[3]

	o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)

	model = Model( img_input , o )
	# print(model.output_shape)
	# print(model.shape)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	model.summary()
	exit(0)
	return model, exp_model



if __name__ == '__main__':
	m = FCN8( 101 )
	from keras.utils import plot_model
	plot_model( m , show_shapes=True , to_file='model.png')
