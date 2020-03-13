# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 10:46:05 2018

@author: Giulia.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import sys
import edge_detection_function
import reading_dataset_function


# *******************************************************************

index1= 131
index2= 132
# *******************************************************************

#folder_path = "AE4317_2019_datasets/sim_poles/20190121-160844/"
folder_path = "AE4317_2019_datasets/sim_poles_panels/20190121-161422/"
#folder_path = "AE4317_2019_datasets/sim_poles_panels_mats/20190121-161931/"

images = reading_dataset_function.reading_dataset(folder_path)

# images = reading_dataset_function.RGB2BGR(images)

start = time.process_time()

edges_old = edge_detection_function.edge_detection(images[index1])
edges_new = edge_detection_function.edge_detection(images[index2])


feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.03,
                       minDistance = 2,
                       blockSize = 2 )
    
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (30,30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# detect features:
points_old = cv2.goodFeaturesToTrack(edges_old, mask = None, **feature_params);

# calculate optical flow
points_new, status, error_match = cv2.calcOpticalFlowPyrLK(edges_old, edges_new, points_old, None, **lk_params)


# filter the points by their status:
points_old = points_old[status == 1];
points_new = points_new[status == 1];

flow_vectors = points_new - points_old;

print('Total time with edge detection = ', time.process_time() - start)


graphics= True

if(graphics):
    im = (0.5 * images[index1].copy().astype(float) + 0.5 * images[index2].copy().astype(float)) / 255.0;
    n_points = len(points_old);
    color = (0.0,1.0,0.0);
    for p in range(n_points):
        cv2.arrowedLine(im, tuple(points_old[p, :]), tuple(points_new[p,:]), color);
     
    plt.figure();
    plt.imshow(im);
    plt.title('Optical flow from edges');
    
    