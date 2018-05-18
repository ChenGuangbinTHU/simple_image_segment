import argparse
import LoadBatches
import FCN8
import FCN_Atrous
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import metrics
import deeplabv3


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )
parser.add_argument("--n_classes", type=int )
parser.add_argument("--input_height", type=int , default =512  )
parser.add_argument("--input_width", type=int , default = 512 )

parser.add_argument('--validate',action='store_false', default=True)
parser.add_argument("--val_images", type = str , default = "")
parser.add_argument("--val_annotations", type = str , default = "")

parser.add_argument("--epochs", type = int, default = 2 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str , default = "")

parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

# modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = FCN_Atrous.FCN8_Atrous
# modelFN = FCN8.FCN8
modelFN = deeplabv3.deeplabv3_plus

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='binary_crossentropy',
      optimizer= optimizer_name ,
      metrics=[ 'accuracy'])

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
# m_exp.compile(loss='binary_crossentropy',optimizer=optimizer_name,metrics=['accuracy'])


if len( load_weights ) > 0:
	m.load_weights(load_weights)


print ("Model output shape" ,  m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth
print(output_height)
print(output_width)


train_x, train_y, train_y_exp = LoadBatches.get_x_and_y(train_images_path, train_segs_path, 'exception_train', n_classes, input_height, input_width,output_height, output_width)
# G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


checkpoint = ModelCheckpoint(save_weights_path+'.0', monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
# checkpoint_exp = ModelCheckpoint(save_weights_path+'exp', monitor='val_acc', verbose=1, save_best_only=True,mode='max')

if validate:
	val_x, val_y, val_y_exp = LoadBatches.get_x_and_y(val_images_path, val_segs_path, 'exception_val', n_classes, input_height, input_width,output_height, output_width)
	# G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )
# print(val_y_exp)
# exit(0)
if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		# m.save_weights( save_weights_path + "." + str( ep ) )
		# m.save( save_weights_path + ".model." + str( ep ) )
else:
	# for ep in range( epochs ):
	# 	print(ep)
		# m.fit_generator( G , 13  ,shuffle = False, validation_data=G2 , validation_steps=5 ,class_weight=[1.0, 1.0],  epochs=1, callbacks=callbacks_list )
	m.fit(train_x, train_y, batch_size=5, epochs=epochs, validation_data=(val_x, val_y), callbacks=callbacks_list)
	# m_exp.fit(train_x, train_y_exp, batch_size=10, epochs=100, validation_data=(val_x, val_y_exp), callbacks=[checkpoint_exp])
		# m.save_weights( save_weights_path + "." + str( ep )  )
		# m.save( save_weights_path + ".model." + str( ep ) )


