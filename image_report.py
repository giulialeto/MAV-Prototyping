#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:23:00 2020

@author: Giulia
"""

#Program to color a certain part of the image in color and the rest in grayscale

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

folder_path = "82614000.jpg"
colored_image = cv2.imread( folder_path )
#colored_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR);
gray = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY);

h,  w = colored_image.shape[:2]

#Define a new image

pixels= np.zeros((h,w,3), dtype="uint8")

#If x and y are in the given range, use colored picture, otherwise grayscale

for y in range(int(w)):
    for x in range(int(h)):
        if x >= h/2 and x<= h and y>=w/4 and y<= w/4*3: 
            pixels[x,y]= colored_image[x,y]
        else:
            pixels[x,y]= gray[x,y]

cv2.imwrite('Green_detection.png',pixels)

#SAME BUT FOR ORANGE DETECTION

folder_path = "999999999.jpg"
colored_image = cv2.imread( folder_path )
#colored_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR);
gray = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY);

h,  w = colored_image.shape[:2]

pixels= np.zeros((h,w,3), dtype="uint8")

#If x and y are in the given range, use colored picture, otherwise grayscale

for y in range(int(w)):
    for x in range(int(h)):
        if x >= h/100*12 and x<= h and y>=w/8 and y<= w/8*7: 
            pixels[x,y]= colored_image[x,y]
        else:
            pixels[x,y]= gray[x,y]

#Detection on the sides

folder_path = "552679000.jpg"
colored_image = cv2.imread( folder_path )
#colored_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2BGR);
gray = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY);

h,  w = colored_image.shape[:2]

pixels= np.zeros((h,w,3), dtype="uint8")

#If x and y are in the given range, use colored picture, otherwise grayscale

for y in range(int(w)):
    for x in range(int(h)):
        if x >= h/2 and x<= h and y>=0 and y<= w/4: 
            pixels[x,y]= colored_image[x,y]
        elif x >= h/2 and x<= h and y>=3*w/4 and y<= w: 
            pixels[x,y]= colored_image[x,y]
        else:
            pixels[x,y]= gray[x,y]


plt.imshow(pixels)
pl.pause(1)

cv2.imwrite('Side_detection.png',pixels)