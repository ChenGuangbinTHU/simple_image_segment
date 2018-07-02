train_images = 'new_sheep_image/' #folder to save images for training
train_annotations = 'new_sheep_seg/' #folder to save ground truth for training
val_images = 'new_sheep_val_image/'
val_annotations = 'new_sheep_val_seg/'

save_weights_path = 'w/' #folder to save best weights

n_classes = 2 #label nums {background and sheep currently}

input_height = 300
input_width = 300

epochs = 100
batch_size = 5
val_batch_size = 2
load_weights = ''  # train from trained weights
model_name = 'resnet_aspp' # choose from {resnet_aspp, fcn8, vgg16_aspp}
optimizer_name = 'adadelta'
save_model_name = 'best_model'

test_images = 'new_sheep_val_image/' #folder to save images that need to be predicted
test_annotations = 'new_sheep_val_seg/'

test_model_name = 'best_model' # model weights in [save_weights_path] used to predict
output_path = 'predict_target/' #folder to save predict results

use_gpu = False

visulize = True # whether to visulize result by using image and corresponding predicting result
visulize_image_path = 'result/' #folder to save visulized results

show_iou = False # whether to calculate iou

#no need to modify below code
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