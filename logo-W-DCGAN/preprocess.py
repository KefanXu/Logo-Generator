from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def resize_logo():
    dirs = os.listdir('birds')
    imgs = [Image.open(os.path.join('birds', img)).resize((64, 64)).save(os.path.join('birds_resize_64', img)) for img in dirs]


def read_logo(logo_path='logo_resize_64'):
    dirs = os.listdir(logo_path)
    dirs = [os.path.join(logo_path, img) for img in dirs]
    imgs = [np.asarray(Image.open(img)) for img in dirs]
    imgs = np.asarray(imgs, dtype=np.int)
    imgs = (imgs/127.5) - 1
    return imgs


if __name__ == '__main__':
    resize_logo()




