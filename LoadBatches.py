from PIL import Image
import numpy as np
import cv2
import glob
import itertools
import random


def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):

	try:
		img = cv2.imread(path, 1)

		if imgNorm == "sub_and_divide":
			img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
		elif imgNorm == "sub_mean":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img[:,:,0] -= 103.939
			img[:,:,1] -= 116.779
			img[:,:,2] -= 123.68
		elif imgNorm == "divide":
			img = cv2.resize(img, ( width , height ))
			img = img.astype(np.float32)
			img = img/255.0

		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img
	except Exception as e:
		print (path , e)
		img = np.zeros((  height , width  , 3 ))
		if odering == 'channels_first':
			img = np.rollaxis(img, 2, 0)
		return img





def getSegmentationArr( path , nClasses ,  width , height  ):

	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		img = cv2.resize(img, ( width , height ))
		img = img[:, : , 0]

		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print (e)
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels

'''
def getSegmentationArr( path , nClasses ,  width , height  ):
		
	seg_labels = np.zeros((  height , width  , nClasses ))
		
	try:
		img = load_label(path)

		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)

	except Exception as e:
		print (e)
		
	seg_labels = np.reshape(seg_labels, ( width* height , nClasses ))
	print(seg_labels)
	exit(0)
	# print(seg_labels.shape)
	# exit(0)
	return seg_labels

def load_label(img_path):
	"""
	Load label image as 1 x height x width integer array of label indices.
	The leading singleton dimension is required by the loss.
	"""
	im = np.array(Image.open(img_path)).astype(np.float32)
	img_h, img_w = im.shape
	pad_h = 224 - img_h
	pad_w = 224 - img_w
	im = np.pad(im, pad_width=((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
	label = np.array(im, dtype=np.uint8)
	# print(label)
	return label
'''

def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
	
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'

	images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
	segmentations.sort()
	# print(images)
	# print(segmentations)
	# exit(0)
	assert len( images ) == len(segmentations)
	for im , seg in zip(images,segmentations):
		assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

	zipped = itertools.cycle( zip(images,segmentations) )
	index = [i for i in range(0, len(images))]
	while True:
		random.shuffle(index)
		# print(index)
		images_ = [images[i] for i in index]
		segmentations_ = [segmentations[i] for i in index]
		# print(images_)
		# print(segmentations_)
		# exit(0)
		zipped = itertools.cycle( zip(images_,segmentations_) )
		X = []
		Y = []
		for _ in range( batch_size) :
			im , seg = next(zipped)
			# print(im, seg)
			X.append( getImageArr(im , input_width , input_height )  )
			Y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )
		# print(1)
		yield np.array(X) , np.array(Y)


# import Models , LoadBatches
# G  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_train/" ,  "data/clothes_seg/prepped/annotations_prepped_train/" ,  1,  10 , 800 , 550 , 400 , 272   ) 
# G2  = LoadBatches.imageSegmentationGenerator( "data/clothes_seg/prepped/images_prepped_test/" ,  "data/clothes_seg/prepped/annotations_prepped_test/" ,  1,  10 , 800 , 550 , 400 , 272   ) 

# m = Models.VGGSegnet.VGGSegnet( 10  , use_vgg_weights=True ,  optimizer='adadelta' , input_image_size=( 800 , 550 )  )
# m.fit_generator( G , 512  , nb_epoch=10 )


