#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:48:22 2020

@author: Giulia
"""

import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def reading_dataset(folder_path):
    
    """
    INPUTs folder path in the format
    
    folder_path = "AE4317_2019_datasets/cyberzoo_poles/20190121-135009/"
    
    OUTPUTs
    images: array cotaining the RGB coordinates of all pictures in the folder with the correct order
    
    """
    
    #Find out how many figures are in the sequence
    path, dirs, files = next(os.walk( folder_path ))
    
    #Inizialize empty array for images
    images = np.empty(len( files ), dtype=object)
    
    #Load all images in the correct sequence
    i= 0
    for image in sorted(os.listdir(folder_path)):
        images[i] = cv2.imread( folder_path+image )
        i= i+1
    
    return images;

def RGB2BGR(images):
    
    """
    INPUTs
    images: array containing the RGB coordinates of a sequence of images
    
    OUTPUTs
    images: array cotaining the BGR coordinates of a sequence of images
    """
    
    i=0
    for image in images:
        images[i] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        i=i+1
    return images;

def show_sequence(images):
    #Show sequence of images
    for image in images:
        plt.imshow(image)
        pl.pause(.0001)
    return images;
    
folder_path = "AE4317_2019_datasets/cyberzoo_poles/20190121-135009/"

images= reading_dataset(folder_path)

images= RGB2BGR(images)

show_sequence(images)