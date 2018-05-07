from PIL import Image
import numpy as np

def visulize(img, mask):
    img = Image.open(img)
    mask = Image.open(mask)
    # img = np.array(img)[0:1080,400:1480]
    # mask = np.array(mask)[:,:,0]
    # img = Image.fromarray(img)
    mask_data = mask.getdata()
    a = img.getdata()
    img = img.convert('RGBA')
    l = list()
    for i,j in zip(mask_data,a):
        if i[0] == 1 and i[1] == 1 and i[2] == 1:
            l.append(( 255, j[1], j[2], 150))
        else:
            l.append((j[0], j[1], j[2], 255))
    img.putdata(l)
    img.show()

def main():
    visulize('new_sheep_val_image/03.jpg','new_sheep_val_seg/03.png')
    # transPNG('new_sheep_val_seg/03.png','1.png')

if __name__ == '__main__':
    main()

