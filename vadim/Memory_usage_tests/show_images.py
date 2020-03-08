import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle
from shutil import copyfile

import subprocess
import sys
import json
import glob

# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 


fx_list = [21,31,61,81,101,121,152]
fy_list = [21,31,61,81,101,121,252]
fz_list = [21,31,61,67,67,67,67]

origin_list =  [[0.2, 0.2, 0.91],\
        [0.3, 0.3, 1.34333],\
        [0.6, 0.6, 2.64333],\
        [0.8, 0.8, 2.9033],\
        [1, 1, 2.90333],\
        [1.2, 1.2, 2.90333],\
        [1.5, 2.5, 2.90333]]

fov_list = [85.4187799, 105.4187799, 85.41877, 96.27356, \
       108.59441, 118.08409, 147.86411]

lookat_list = [[0.21, 0.21, 0.56],\
        [0.31, 0.31, 0.826667],\
        [0.61, 0.61, 1.62667],\
        [0.81, 0.81, 1.78667],\
        [1.01, 1.01, 1.78667],\
        [1.21, 1.21, 1.78667],\
        [1.52, 2.52, 1.78667]]
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
start_from_index = 7
if(0):    
    for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
        
        loop_index = fx_list.index(fx)
        if(start_from_index<=loop_index):
            continue
        
        
        print( "--{}x{}x{}--".format(fx,fy,fz))
        print( "njobs = 72")
        output_dir = '../experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
        path = '/home/vhold/pyshdom/notebooks/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
        cam_nx = fx
        cam_ny = fy
        
        # load the measurments to see the rendered images:
        medium, solver, measurements = shdom.load_forward_model(output_dir)
        # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
        USED_CAMERA = measurements.camera
        RENCERED_IMAGES = measurements.images
        f, axarr = plt.subplots(1, 2, figsize=(20, 20))
        ax = axarr.ravel()
        original_image = RENCERED_IMAGES[0]
        gamma_image = (original_image/np.max(original_image))**0.5
        
        ax[0].imshow(original_image,cmap='gray')
        ax[0].invert_xaxis() 
        ax[0].invert_yaxis() 
        ax[0].axis('off')
        ax[0].set_title("tamar_cvpr_rico{}x{}x{}.txt".format(fx,fy,fz))
        
        ax[1].imshow(gamma_image,cmap='gray')
        ax[1].invert_xaxis() 
        ax[1].invert_yaxis() 
        ax[1].axis('off')
        ax[1].set_title('one view gamma corrected')


# --------------------------------------
f, axarr = plt.subplots(3, 3, figsize=(20, 20))
axarr = axarr.ravel()

for ax,fx,fy,fz,origin,fov,lookat in zip(axarr,fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
    
    loop_index = fx_list.index(fx)
    if(start_from_index<=loop_index):
        continue
    
    
    print( "--{}x{}x{}--".format(fx,fy,fz))
    print( "njobs = 72")
    output_dir = '../experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
    path = '/home/vhold/pyshdom/notebooks/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
    cam_nx = fx
    cam_ny = fy
    
    # load the measurments to see the rendered images:
    measurements = shdom.load_forward_model_measurements(output_dir)
    # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    USED_CAMERA = measurements.camera
    RENCERED_IMAGES = measurements.images
    original_image = RENCERED_IMAGES[0]
    gamma_image = (original_image/np.max(original_image))**0.5
    
    ax.imshow(original_image,cmap='gray')
    ax.invert_xaxis() 
    ax.invert_yaxis() 
    ax.axis('off')
    ax.set_title("tamar_cvpr_rico{}x{}x{}.txt".format(fx,fy,fz))
    
    
    
plt.show()