from PIL import Image
import numpy as np
import os

def visulize(img, mask):
    save_name = mask.split('/')[1].split('.')[0]
    print(save_name)
    img = Image.open(img)
    shape = np.array(img).shape
    # print(shape)
    mask = Image.open(mask).resize((shape[0],shape[0]))
    mask = np.array(mask)[:,:,0]
    mask[mask < 250] = 0
    mask[mask >= 250] = 1
    mask = Image.fromarray(mask)
    mask = np.lib.pad(mask, ((0,0),(400,shape[1]-shape[0]-400)), 'constant', constant_values=(0, 0))
    # print(np.array(img).shape)
    # print(np.array(mask).shape)
    # img = np.array(img)[0:1080,400:1480]
    # mask = np.array(mask)[:,:,0]
    # img = Image.fromarray(img)
    mask = Image.fromarray(mask)
    mask_data = mask.getdata()
    a = img.getdata()
    img = img.convert('RGBA')
    l = list()
    for i,j in zip(mask_data,a):
        # print(i)
        if i == 1:
            l.append(( 255, j[1], j[2], 150))
        else:
            l.append((j[0], j[1], j[2], 255))
    img.putdata(l)
    img.save('result/'+save_name+'.png')

def main():
    imgs_folder = 'new_sheep_val_image/'
    mask_folder = 'predict_target/'
    imgs = os.listdir(imgs_folder)
    masks = os.listdir(mask_folder)
    imgs.sort()
    masks.sort()
    assert(len(imgs) == len(masks))
    for img,mask in zip(imgs,masks):
        visulize(imgs_folder+img, mask_folder+mask)
        # exit(0)

    # visulize('new_sheep_val_image/03.jpg','new_sheep_val_seg/03.png')
    # transPNG('new_sheep_val_seg/03.png','1.png')

if __name__ == '__main__':
    main()

