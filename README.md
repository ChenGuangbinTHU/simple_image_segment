# README

simply implement image segmentation for my own project(extracting sheep from iamge) , mainly reference deeplab series and fcn(fully conv network)

## experiment environment
ubuntu 17.04 + keras + cuda9 + cudnn7

## how to use

1. clone this responsity to local
2. cd simple_image_segment
3. pip install -r requirement.txt (this will autoly install tensorflow **gpu** version)
4. modify config.py based on your own need
5. python train.py
6. python predict.py
