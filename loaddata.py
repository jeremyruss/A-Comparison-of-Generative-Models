import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imageio
import random

datadir = './images/'
categories = ['amulets', 'axes', 'bodies', 'boots', 'bows', 'capes', 'gloves',
              'helms', 'legs', 'rings', 'shields', 'staffs', 'swords']

def training_data():
    x = []
    y = []
    for i, category in enumerate(categories):
        path = os.path.join(datadir, category)
        for image in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (32,32))
                image = image / 255.0
                x.append(image)
                y.append(i)
            except Exception as e:
                pass
    x_train = np.asarray(x, dtype='float32')
    y_train = np.asarray(y, dtype='float32')
    np.save(os.path.join(datadir, 'x_train.npy'), x_train)
    np.save(os.path.join(datadir, 'y_train.npy'), y_train)
    print(x_train.shape, y_train.shape)

def load_data():
    x_train = np.load('./images/x_train.npy', allow_pickle=True)
    y_train = np.load('./images/y_train.npy', allow_pickle=True)
    print(x_train.shape, y_train.shape)
    return x_train, y_train

def save_gif(datadir, name):
    images = []
    for filename in os.listdir(datadir):
        if filename.endswith('.png'):
            path = os.path.join(datadir, filename)
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (300, 300))
            images.append(rgb)
    imageio.mimsave(os.path.join('./images/', name), images)

def save_rgb(datadir):
    for filename in os.listdir(datadir):
        if filename.endswith('.png'):
            path = os.path.join(datadir, filename)
            rgb_path = os.path.join(datadir, 'rgb/', filename)
            bgr = cv2.imread(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(rgb_path, rgb)

def save_training_example():
    n = 0
    fig = plt.figure(figsize=(8, 8))
    while n < 64:
        for category in categories:
            if n == 64:
                break
            path = os.path.join(datadir, category)
            image = random.choice(os.listdir(path))
            while not image.endswith('.jpg'):
                image = random.choice(os.listdir(path))
            try:
                image = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                image = cv2.resize(image, (32, 32))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                continue
            plt.subplot(8, 8, n + 1)
            plt.imshow(image)
            plt.axis('off')
            n += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig('./img/examples.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    #save_gif('./output/cvae/', 'cvae.gif')
    #save_gif('./output/dcgan/', 'dcgan.gif')
    save_training_example()
