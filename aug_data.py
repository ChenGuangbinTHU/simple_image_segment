from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array  

data_gen_x = ImageDataGenerator(rotation_range=40,  
                              width_shift_range=0.2,  
                              height_shift_range=0.2,  
                              horizontal_flip=True,  
                              vertical_flip=True,  
                              fill_mode='nearest',  
                              data_format='channels_first',
                              )

data_gen_y = ImageDataGenerator(rotation_range=40,  
                              width_shift_range=0.2,  
                              height_shift_range=0.2,  
                              horizontal_flip=True,  
                              vertical_flip=True,  
                              fill_mode='nearest',  
                              data_format='channels_first',
                              )

x = load_img('new_sheep_image/_11.jpg')
y = load_img('new_sheep_seg/_11.png')
x = img_to_array(x, data_format='channels_first')
y = img_to_array(y, data_format='channels_first')
x=x.reshape((1,) + x.shape) 
y=y.reshape((1,) + y.shape) 

i = 0
for batch in data_gen_x.flow(x,batch_size=1,  
                    save_to_dir='aug_data/train/x/',  
                    save_prefix='next_gen',  
                    save_format='jpeg',seed=1):
    i += 1
    print(i)
    if i > 20:
        break

j = 0
for batch in data_gen_y.flow(y,batch_size=1,  
                    save_to_dir='aug_data/train/y/',  
                    save_prefix='next_gen',  
                    save_format='jpeg',seed=1):
    j += 1
    print(j)
    if j > 20:
        break
