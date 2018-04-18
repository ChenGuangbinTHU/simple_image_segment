from PIL import Image
import os
import numpy as np
import cv2

pre_folder = 'pre_sheep_seg/'
save_folder = 'sheep_seg/'

files = os.listdir(pre_folder)
files.sort()

for f in files:
    img = np.array(Image.open(pre_folder+f)).astype(np.uint8)
    height, width = img.shape
    new_img = np.zeros((height, width, 3)).astype(np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i, j] == 1:
                new_img[i,j] = 1
    cv2.imwrite( save_folder + f ,new_img )
    # exit(0)