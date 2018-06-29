from PIL import Image
import numpy as np
import os

def visulize(img, mask, save_folder):
    save_name = mask.split('/')[1].split('.')[0]
    print(save_name)
    img = Image.open(img)
    shape = np.array(img).shape
    # print(shape)
    mask = Image.open(mask).resize((shape[0],shape[0]))
    mask = np.array(mask)[:,:,0]
    mask[mask < 250] = 0
    mask[mask >= 250] = 1
    mask = np.lib.pad(mask, ((0,0),(400,shape[1]-shape[0]-400)), 'constant', constant_values=(0, 0))
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
    img.save(save_folder+save_name+'.png')


def main(imgs_folder, mask_folder, save_folder):
    imgs = os.listdir(imgs_folder)
    masks = os.listdir(mask_folder)
    imgs.sort()
    masks.sort()
    assert(len(imgs) == len(masks))
    for img,mask in zip(imgs,masks):
        print(img)
        visulize(imgs_folder+img, mask_folder+mask, save_folder)
        # exit(0)

    # visulize1('new_sheep_val_image/03.jpg','predict_target/03.jpg')
    # transPNG('new_sheep_val_seg/03.png','1.png')

if __name__ == '__main__':
    main()

