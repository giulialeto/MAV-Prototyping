#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:53:15 2020

@author: Giulia
"""


import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import time


folder_path = "AE4317_2019_datasets/cyberzoo_poles/20190121-135009/"

#Find out how many figures are in the sequence
path, dirs, files = next(os.walk( folder_path ))

#Inizialize empty array for images
images = np.empty(len( files ), dtype=object)

#Load all images in the correct sequence and convert colors
i= 0
for image in sorted(os.listdir(folder_path)):
    images[i] = cv2.imread( folder_path+image )
    #images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
    i= i+1

##Show sequence of images
#for i in range(0, len(files))
#    plt.imshow(images[i])
#    pl.pause(.0001)

##============================== Find edges with Canny ==========================
#select image for which you want to find the boundaries
k= 200

start = time.process_time()

edges = cv2.Canny(images[k],100,200)

print(time.process_time() - start)


##============================== Show output of Canny ==========================
plt.subplot(121),plt.imshow(images[k],cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
