from PIL import Image
import numpy as np
import cv2
import glob
import itertools
import random
import pickle
import keras
# from matplotlib import pyplot as plt

def getImageArr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):
	# print(path)
	try:
		img = cv2.imread(path, 1)
		img = img[0:1080,400:1480]
		img = cv2.resize(img, ( width , height ))
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
	# print(path)
	seg_labels = np.zeros((  height , width  , nClasses ))
	try:
		img = cv2.imread(path, 1)
		# print(img.shape)
		img = img[0:1080,400:1480]
		img = cv2.resize(img, ( width , height ))
		img = img[:, : , 0]
		for c in range(nClasses):
			seg_labels[: , : , c ] = (img == c ).astype(int)
		# Image.fromarray(img).show()
	except Exception as e:
		print (e)
		
	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	# seg_labels = seg_labels.transpose((2,0,1))
	return seg_labels


def get_x_and_y(img_path, seg_path, n_classes, input_height, input_width, output_width, output_height):
	assert img_path[-1] == '/'
	assert seg_path[-1] == '/'
	# assert exp_path[-1] == '/'
	x = []
	y = []
	y_exception = []
	
	images = glob.glob( img_path + "*.jpg"  ) + glob.glob(img_path +  "*.png"  ) +  glob.glob( img_path + "*.jpeg"  )
	images.sort()
	segmentations  = glob.glob( seg_path + "*.jpg"  ) + glob.glob( seg_path + "*.png"  ) +  glob.glob( seg_path + "*.jpeg"  )
	segmentations.sort()
	assert len( images ) == len(segmentations)
	
	index = [i for i in range(0, len(images))]
	random.shuffle(index)
	images = [images[i] for i in index]
	segmentations = [segmentations[i] for i in index]
	
	for img, seg in zip(images, segmentations):
		x.append( getImageArr(img , input_width , input_height )  )
		y.append( getSegmentationArr( seg , n_classes , output_width , output_height )  )

	return np.array(x), np.array(y)
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
'''
if __name__ == '__main__':
	getSegmentationArr('aug_data/train/y/01.png', 2, 600, 600)