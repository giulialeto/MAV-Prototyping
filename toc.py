# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:19:58 2020

@author: lujingyi
"""

import os
from os.path import join
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import extract_information_flow_field_copy as OF

folder_path = "20200318-181952/"

#Find out how many figures are in the sequence
path, dirs, files = next(os.walk( folder_path ))

#Inizialize empty array for images
images = np.empty(len( files ), dtype=object)

#Load all images in the correct sequence and convert colors
i= 0
filenames = os.listdir(folder_path)
filenames.sort(key=lambda x:int(x[:-4]))
for image in filenames:
    images[i] = cv2.imread( folder_path+image )
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
    i= i+1
    
#img = images[543]
#plt.imshow(img),plt.show()

errtab=[]
toctab=[]
for i in range(543,len(images)-1): #only testing images with trees
    index1 = i
    index2 = i+1
    points_old, points_new, flow_vectors = OF.show_flow(index1, index2,images)
    pu, pv, err=OF.estimate_linear_flow_field(points_old, flow_vectors, RANSAC=True, n_iterations=100, error_threshold=10.0)
    errtab.append(err)
    toc = OF.time_to_contact(pu,pv)
    toctab.append(toc)
    
    
#time_start = time.clock()    
#index1 = 544
#index2 = 545
#
#points_old, points_new, flow_vectors = OF.show_flow(index1, index2,images);
#pu, pv, err=OF.estimate_linear_flow_field(points_old, flow_vectors, RANSAC=False, n_iterations=100, error_threshold=10.0)
#toc = OF.time_to_contact(pu,pv)
#print(time.clock() - time_start)
#===============================================================================
#n_points = points_old.shape[0];
#
## make a matrix A with elements [x,1] 
#A_hor = np.concatenate((points_old[:,0].reshape(n_points,1), np.ones([points_old.shape[0], 1])), axis=1);
## Moore-Penrose pseudo-inverse:
## https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
#pseudo_inverse_A_hor = np.linalg.pinv(A_hor);
## target = horizontal flow:
#u_vector = flow_vectors[:,0];
## solve the linear system:
#pu = np.dot(pseudo_inverse_A_hor, u_vector);
## calculate how good the fit is:
#errs_u = np.abs(np.dot(A_hor, pu) - u_vector);
#
## Plot the fit:
#plt.figure();
#plt.plot(points_old[:,0], flow_vectors[:,0], 'x');
#min_x = np.min(points_old[:,0]);
#u_min_x = np.dot(np.asarray([min_x, 1]).reshape(1,2), pu);
#max_x = np.max(points_old[:,0]);
#u_max_x = np.dot(np.asarray([max_x, 1]).reshape(1,2), pu);
#plt.plot([min_x, max_x], [u_min_x, u_max_x], 'r');
#plt.xlabel('x [px]');
#plt.ylabel('u [px]');
#plt.title('Horizontal flow fit')
#
## make a matrix A with elements [y,1]
#A_ver = np.concatenate((points_old[:,1].reshape(n_points,1), np.ones([points_old.shape[0], 1])), axis=1);
## Moore-Penrose pseudo-inverse:
## https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
#pseudo_inverse_A_ver = np.linalg.pinv(A_ver);
## target = vertical flow:
#v_vector = flow_vectors[:,1];
#pv = np.dot(pseudo_inverse_A_ver, v_vector);
#errs_v = np.abs(np.dot(A_ver, pv) - v_vector);
#err = (np.mean(errs_u) + np.mean(errs_v)) / 2.0;
#
## Plot the fit:
#plt.figure();
#plt.plot(points_old[:,1], flow_vectors[:,1], 'x');
#min_y = np.min(points_old[:,1]);
#v_min_y = np.dot(np.asarray([min_y, 1]).reshape(1,2), pv);
#max_y = np.max(points_old[:,1]);
#v_max_y = np.dot(np.asarray([max_y, 1]).reshape(1,2), pv);
#plt.plot([min_y, max_y], [v_min_y, v_max_y], 'r');
#plt.xlabel('y [px]');
#plt.ylabel('v [px]');
#plt.title('Vertical flow fit');
#
#
#print('pu = {} (W/Z), {} (-U/Z)'.format(pu[0], pu[1]))
#print('Error u = {}, error v = {}'.format(np.mean(errs_u), np.mean(errs_v)))
#===============================================================================
#n_points = points_old.shape[0];
#
##RANSAC parameters - change to do the exercise:
#n_iterations=50;
#error_threshold=10.0;
#sample_size = 3; # 3 is the minimum for this fit. You can set it higher, though.
#
# # This is a RANSAC method to better deal with outliers
## matrices and vectors for the big system:
#A = np.concatenate((points_old, np.ones([points_old.shape[0], 1])), axis=1);
#u_vector = flow_vectors[:,0];
#v_vector = flow_vectors[:,1];
#
## solve many small systems, calculating the errors:
#errors = np.zeros([n_iterations, 2]);
#pu = np.zeros([n_iterations, 3])
#pv = np.zeros([n_iterations, 3])
#for it in range(n_iterations):
#    inds = np.random.choice(range(n_points), size=sample_size, replace=False);
#    AA = np.concatenate((points_old[inds,:], np.ones([sample_size, 1])), axis=1);
#    pseudo_inverse_AA = np.linalg.pinv(AA);
#    # horizontal flow:
#    u_vector_small = flow_vectors[inds, 0];
#    # pu[it, :] = np.linalg.solve(AA, UU);
#    pu[it,:] = np.dot(pseudo_inverse_AA, u_vector_small);
#    errs = np.abs(np.dot(A, pu[it,:]) - u_vector);
#    errs[errs > error_threshold] = error_threshold;
#    errors[it, 0] = np.mean(errs);
#    # vertical flow:
#    v_vector_small = flow_vectors[inds, 0];
#    # pv[it, :] = np.linalg.solve(AA, VV);
#    pv[it, :] = np.dot(pseudo_inverse_AA, v_vector_small);
#    errs = np.abs(np.dot(A, pv[it,:]) - v_vector);
#    errs[errs > error_threshold] = error_threshold;
#    errors[it, 1] = np.mean(errs);
#
## take the minimal error
#errors = np.mean(errors, axis=1);
#ind = np.argmin(errors);
#err = errors[ind];
#pu = pu[ind, :];
#pv = pv[ind, :];
#
## Plot the fit:
#pu = np.asarray([pu[0], pu[2]]).reshape(2,1);
#plt.figure();
#plt.plot(points_old[:,0], flow_vectors[:,0], 'x');
#min_x = np.min(points_old[:,0]);
#u_min_x = np.dot(np.asarray([min_x, 1]).reshape(1,2), pu);
#u_min_x = u_min_x[0][0];
#max_x = np.max(points_old[:,0]);
#u_max_x = np.dot(np.asarray([max_x, 1]).reshape(1,2), pu);
#u_max_x = u_max_x[0][0];
#plt.plot([min_x, max_x], [u_min_x, u_max_x], 'r');
#plt.xlabel('x [px]');
#plt.ylabel('u [px]');
#plt.title('Horizontal flow fit')
#
## Plot the fit:
#pv = np.asarray([pv[0], pv[2]]).reshape(2,1);
#plt.figure();
#plt.plot(points_old[:,1], flow_vectors[:,1], 'x');
#min_y = np.min(points_old[:,1]);
#v_min_y = np.dot(np.asarray([min_y, 1]).reshape(1,2), pv);
#v_min_y = v_min_y[0][0];
#max_y = np.max(points_old[:,1]);
#v_max_y = np.dot(np.asarray([max_y, 1]).reshape(1,2), pv);
#v_max_y = v_max_y[0][0];
#plt.plot([min_y, max_y], [v_min_y, v_max_y], 'r');
#plt.xlabel('y [px]');
#plt.ylabel('v [px]');
#plt.title('Vertical flow fit');