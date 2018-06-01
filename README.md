train:
python train.py --save_weights_path='w/'  --train_images='new_sheep_image/' --train_annotations='new_sheep_seg/' --n_classes=2  --model_name='fcn8' --val_images='new_sheep_val_image/' --val_annotations='new_sheep_val_seg/' --epochs=100

predict:
python predict.py --save_weights_path='w/' --epoch_number=0 --test_images='new_sheep_val_image/' --output_path='predict_target/' --n_classes=2  --model_name='FCN8'

to do:same-sized predict mask to image 
        resnet
        upsample logits
        data augmentation