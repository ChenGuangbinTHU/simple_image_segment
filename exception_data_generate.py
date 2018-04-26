import pickle
import os
from PIL import Image

def main():
    img_path = 'new_sheep_val_image/'
    d = {}
    f = open('exception_val', 'wb')
    imgs = os.listdir(img_path)
    imgs.sort()
    for img in imgs:
        # print(img)
        Image.open(img_path+img).show()
        x = input()
        d[img] = x
        
        # print(d)
    pickle.dump(d, f)
    f.close()

if __name__ == '__main__':
    # main()
    f = open('exception_val', 'rb')
    a = pickle.load(f)
    print(a)