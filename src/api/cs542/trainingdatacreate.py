import numpy as np
import cv2
import random
from PIL import Image
import os.path
from pathlib import Path

oripath = "../../../data/input/License/Train"
noblurpath = "../../../data/output/cs542/s_cnn/train/no_blur/"
blurpath = "../../../data/output/cs542/s_cnn/train/blur/"
inputpath = "../../../data/output/cs542/s_cnn/train/inputdata/"

for path in [noblurpath, blurpath, inputpath]:
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True)

gridx=30
gridy=30

#go through every image in source folder
print('begin loading images')
good_imgs = []
bad_imgs = []
valid_images = [".jpg"]

for f in os.listdir(os.path.join(oripath, 'Good_License')):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    good_imgs.append(Image.open(os.path.join(oripath, 'Good_License', f)))

for f in os.listdir(os.path.join(oripath, 'Bad_License')):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    bad_imgs.append(Image.open(os.path.join(oripath, 'Bad_License', f)))

print('finished loading images')

for i in range(len(good_imgs)):
    #Slicing the pictures first
    img = good_imgs[i]
    (imageWidth, imageHeight) = img.size
    rangex = img.width // gridx
    rangey = img.height // gridy
    for x in range(rangex):
        for y in range(rangey):

            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = img.crop(bbox)
            #In order to make sure the raondom 50% chance of getting blur and noblur images, I'm using random to decide whether do motion blur or not
            #not do motion blur
            slice_bit.save(noblurpath + 'noblur,' +str(i)+'_'+ str(x) + '_' + str(y) + '.jpg', optimize=True, bits=6)
            slice_bit.save(inputpath + 'noblur,' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', optimize=True,bits=6)
            print(str(i))


for i in range(len(bad_imgs)):
    #Slicing the pictures first
    img = bad_imgs[i]
    (imageWidth, imageHeight) = img.size
    rangex = img.width / gridx
    rangey = img.height / gridy
    for x in range(rangex):
        for y in range(rangey):

            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = img.crop(bbox)
            #In order to make sure the raondom 50% chance of getting blur and noblur images, I'm using random to decide whether do motion blur or not
            #not do motion blur
            slice_bit.save(blurpath + 'noblur,' +str(i)+'_'+ str(x) + '_' + str(y) + '.jpg', optimize=True, bits=6)
            slice_bit.save(inputpath + 'noblur,' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', optimize=True,bits=6)
            print(str(i))

















