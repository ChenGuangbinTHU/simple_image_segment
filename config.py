save_weights_path = 'w/'
train_images = 'new_sheep_image/'
train_annotations = 'new_sheep_seg/'
n_classes = 2
input_height = 300
input_width = 300
val_images = 'new_sheep_val_image/'
val_annotations = 'new_sheep_val_seg/'
epochs = 100
batch_size = 5
val_batch_size = 2
load_weights = ''
model_name = 'resnet_aspp'
optimizer_name = 'adadelta'
save_model_name = 'best_model'

test_images = 'new_sheep_val_image/'
test_annotations = 'new_sheep_val_seg/'
test_model_name = 'best_model'
output_path = 'predict_target/'

use_gpu = False

visulize = True
visulize_image_path = 'result11111/'

show_iou = False

folders = {
    'save_weights_path' : save_weights_path,
    'train_images' : train_images,
    'train_annotations' : train_annotations,
    'val_images' : val_images,
    'val_annotations' : val_annotations,
    'test_images' : test_images,
    'test_annotations' : test_annotations, 
    'output_path' : output_path,
    'visulize_image_path' : visulize_image_path
}
import os

for k,v in folders.items():
    if not v:
        print('please input ' , k ,' in config.py')
    if not os.path.exists(v):
        os.mkdir(v)