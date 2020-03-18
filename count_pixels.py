# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:49:25 2020

@author: lujingyi
"""

import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

folder_path = "AE4317_2019_datasets/sim_poles_panels/20190121-161422/"

#Find out how many figures are in the sequence
path, dirs, files = next(os.walk( folder_path ))

#Inizialize empty array for images
images = np.empty(len( files ), dtype=object)


#YUV threshold for green (obtained from Giulia)
GREEN_MIN = np.array([50, 0, 0], np.uint8)
GREEN_MAX = np.array([200, 120, 135], np.uint8)

#Sort images in sequence
filenames = os.listdir(folder_path)
filenames.sort(key=lambda x:int(x[:-4]))

##==============================implement in C====================================
#Count the number of pixels that satisfies the green threshold
cnt=[] 
i=0
for image in filenames:
    images[i] = cv2.imread( folder_path+image )
    #The original size of the image is [520*240], only the middle bottom of the 
    #image is relevant. Crop the image such that only [130:390 in width, 0:200 
    #in height] will be analyzed. The values can be palyed around for a better
    #performance.
    img = images[i][130:390,0:200]  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)#Convert to YUV
    dst = cv2.inRange(img, GREEN_MIN, GREEN_MAX)#output 0 for pixels without green
    no_green = cv2.countNonZero(dst)#count pixels with green
    cnt.append(no_green)
    i= i+1

#Choose a save cnt threshold
cnt_order=sorted(cnt)
print('Threshold of the number of pixels with green is',cnt_order[1]) #1367

#In the paparazzi code, cv_detect_color_object.c counts the number of pixels with 
#orange. orange_avoider.c asks the UAV to change direction if the number of pixels
#with orange is larger than a threshold. 

#In our case, we count the number of pixels with green. Tell the UAV to change
#direction when cnt is smaller than the threshold.

#The threshold chosen here is based on the simulation performance given by the
#professor. Might need to be adjusted in our own simulation.












##===================================Testing in Python==========================
#plt.imshow(img[160]),plt.show()

#img = images[0]  

#img = images[0][150:370,0:200]


#img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#plt.imshow(img),plt.show()

#Filtered = np.zeros([img.shape[0], img.shape[1]]);
#y_low=50
#y_high=200
#u_low=0
#u_high=120
#v_low=0
#v_high=135
#for y in range(img.shape[0]):
#    for x in range(img.shape[1]):
#        if(img[y,x,0] >= y_low and img[y,x,0] <= y_high and \
#           img[y,x,1] >= u_low and img[y,x,1] <= u_high and \
#           img[y,x,2] >= v_low and img[y,x,2] <= v_high):
#           Filtered[y,x] = 1;
#
#plt.figure()
#plt.imshow(Filtered);
#plt.title('Filtered image');

#GREEN_MIN = np.array([50, 0, 0], np.uint8)
#GREEN_MAX = np.array([200, 120, 135], np.uint8)
#
#dst = cv2.inRange(img, GREEN_MIN, GREEN_MAX)
#no_blue = cv2.countNonZero(dst)
#print('The number of green pixels is: ' + str(no_blue))
#cv2.imshow("opencv",img)
#cv2.waitKey(0)


