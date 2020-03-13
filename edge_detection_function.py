#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:53:15 2020

@author: Giulia
"""

import cv2
import matplotlib.pyplot as plt
import time

##============================== Find edges with Canny ==========================
def edge_detection(image):
    
    start = time.process_time()
    
    edges = cv2.Canny(image,100,200)
    
    print(time.process_time() - start)
    
    # ##============================== Show output of Canny ==========================
    # plt.subplot(121),plt.imshow(images[0],cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    # plt.show()
    
    return(edges)


