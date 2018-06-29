import argparse
import LoadBatches
from models import FCN8, resnet_aspp, vgg16_aspp
import numpy as np
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.layers import *
from keras import metrics
import metrics
from keras import backend as K
import tensorflow as tf
import config

train_images_path = config.train_images
train_segs_path = config.train_annotations
train_batch_size = config.batch_size
n_classes = config.n_classes
input_height = config.input_height
input_width = config.input_width
save_weights_path = config.save_weights_path
epochs = config.epochs
load_weights = config.load_weights

model_name = config.model_name
optimizer_name = config.optimizer_name

val_images_path = config.val_images
val_segs_path = config.val_annotations
val_batch_size = config.val_batch_size

modelFNs = {'fcn8':FCN8.FCN8, 'vgg16_aspp':vgg16_aspp.vgg16_aspp, 'resnet_aspp':resnet_aspp.resnet_aspp}
if model_name not in modelFNs:
	print('please choose model name from {fcn8, vgg16_aspp, resnet_aspp}')
	exit(0)
modelFN = modelFNs[model_name]

NUM_CLASSES = 2


m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='binary_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])



if len( load_weights ) > 0:
	m.load_weights(load_weights)


print ("Model output shape" ,  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth
print(output_height)
print(output_width)


train_x, train_y = LoadBatches.get_x_and_y(train_images_path, train_segs_path, n_classes, input_height, input_width,output_height, output_width)
# G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


checkpoint = ModelCheckpoint(save_weights_path+config.save_model_name, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
# checkpoint_exp = ModelCheckpoint(save_weights_path+'exp', monitor='val_acc', verbose=1, save_best_only=True,mode='max')


val_x, val_y = LoadBatches.get_x_and_y(val_images_path, val_segs_path, n_classes, input_height, input_width,output_height, output_width)


m.fit(train_x, train_y, batch_size=train_batch_size, epochs=epochs, validation_data=(val_x, val_y), callbacks=callbacks_list)



