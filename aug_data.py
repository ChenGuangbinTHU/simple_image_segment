from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array  
from PIL import Image
import numpy as np

data_gen_x = ImageDataGenerator(rotation_range=40,  
                               
                              horizontal_flip=True,  
                              vertical_flip=True,  
                              fill_mode='nearest',  
                              data_format='channels_first',
                              )

data_gen_y = ImageDataGenerator(rotation_range=40,  
                              
                              horizontal_flip=True,  
                              vertical_flip=True,  
                              fill_mode='nearest',  
                              data_format='channels_first',
                              )

def pic_gen(x_file, y_file,postfix):
    x = load_img(x_file)

    y = load_img(y_file)

    x = img_to_array(x, data_format='channels_first')
    y = img_to_array(y, data_format='channels_first')

    # x = x[:,0:1080,400:1480]
    # y = y[:,0:1080,400:1480]
    # Image.fromarray(x).save(x_file)
    # Image.fromarray(y).save(y_file)
    x=x.reshape((1,) + x.shape) 
    y=y.reshape((1,) + y.shape) 

    i = 0
    for batch in data_gen_x.flow(x,batch_size=1,  
                        save_to_dir='aug_data/train/x/',  
                        save_prefix='next_gen_'+str(postfix),  
                        save_format='jpeg',seed=1):
        i += 1
        print('     ', i)
        if i > 5:
            break

    j = 0
    for batch in data_gen_y.flow(y,batch_size=1,  
                        save_to_dir='aug_data/train/y/',  
                        save_prefix='next_gen_'+str(postfix),  
                        save_format='jpeg',seed=1):
        j += 1
        print('     ', j)
        if j > 5:
            break

def process():
    folder = 'aug_data/train/y/'
    files = os.listdir(folder)
    for file in files:
        x = Image.open(folder+file)
        x = np.array(x)
        # print(x.shape)
        x[x[:,:,0] < 220, 0] = 0
        x[x[:,:,0] != 0, 0] = 1
        x[x[:,:,1] < 220, 1] = 0
        x[x[:,:,1] != 0, 1] = 1
        x[x[:,:,2] < 220, 2] = 0
        x[x[:,:,2] != 0, 2] = 1
        Image.fromarray(x).save(folder+file)
        

if __name__ == '__main__':
    import os
    x_path = 'new_sheep_image/'
    y_path = 'new_sheep_seg/'
    x = os.listdir(x_path)
    y = os.listdir(y_path)
    x.sort()
    y.sort()
    num = 0
    for i,j in zip(x,y):
        print(num)
        # print(x,i)
        pic_gen(x_path+i,y_path+j,num)
        num += 1
    process()
