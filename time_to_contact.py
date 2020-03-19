#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:13:35 2020

@author: Giulia
"""

import os
#from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import time
import color_detection_function
import glob

#folder_path = "AE4317_2019_datasets/sim_poles/20190121-160844/"
folder_path = "AE4317_2019_datasets/sim_poles_panels/20190121-161422/"
#folder_path = "AE4317_2019_datasets/sim_poles_panels_mats/20190121-161931/"
#folder_path = "20200310-142119/"


#Find out how many figures are in the sequence
path, dirs, files = next(os.walk( folder_path ))

# #Inizialize empty array for images
images = np.empty(len( files ), dtype=object)

#Load all images in the correct sequence and convert colors
i= 0
for image in sorted(os.listdir(folder_path)):
    color_detection_function.filter_color(folder_path+image,  y_low = 50, y_high = 200, \
                      u_low = 0, u_high = 120, v_low = 0, v_high = 130, resize_factor=1)
    i= i+1
# image = ' 15610000.jpg' #glob.glob('*.jpg')''

# color_detection_function.filter_color(folder_path+image,  y_low = 50, y_high = 200, \
#                       u_low = 0, u_high = 120, v_low = 0, v_high = 130, resize_factor=1)

##Show sequence of images
#for i in range(0, len(files))
#    plt.imshow(images[i])
#    pl.pause(.0001)

# #Show single of images
# plt.imshow(images[0])
# pl.pause(.1)
# ##Show single of images
# plt.imshow(images[70])