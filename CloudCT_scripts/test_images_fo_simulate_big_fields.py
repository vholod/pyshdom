
import os 
import sys
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom 
from shdom import CloudCT_setup
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
from shdom.CloudCT_Utils import *
from collections import OrderedDict
from PIL import Image

# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 


# -----------------------------------------------

CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_256x256x100_5000CCN_50m_micro_256_0000007620.txt"
#CloudFieldFile = "/wdata/vadim/CloudGeometry/clouds from Udi/16samples/BOMEX_512x512x200_20m_20m_1s_512_0000002300_3_3_old_pyshdom.txt"
#CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/danny_small_cloud_old_pyshdom.txt"
#CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_21600_515x515x71_6520_old_pyshdom.txt"


str_name = CloudFieldFile.split('/')[-1].split('.')[0]
file_name = str_name +'.mat'
if not os.path.exists(str_name):
    os.mkdir(str_name)

M = sio.loadmat(file_name)
IMAGES_DICT = OrderedDict()
for satname,img in M.items():
    if 'sat' in satname:
        IMAGES_DICT[satname] = img

names = list(IMAGES_DICT.keys())
Radiances = list(IMAGES_DICT.values())

ncols = 5 # colums, in subplots
nrows_ncols = [int(np.ceil(len(names)/ncols)), int(ncols)]

images_array = np.array(Radiances)
MAXI = images_array.max()
    
fig = plt.figure(figsize=(20, 20))
grid = AxesGrid(fig, 111,
                nrows_ncols=nrows_ncols,
                axes_pad=0.3,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )  

# show all
for projection_index, (ax, name) in enumerate(zip(grid, names)):
    image = IMAGES_DICT[name].copy()
    ax.set_axis_off()
    im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
    ax.set_title("{}".format(name))

grid.cbar_axes[0].colorbar(im)

# new plot one by one:
for projection_index, (ax, name) in enumerate(zip(grid, names)):
    image = IMAGES_DICT[name].copy()
    scale = image.max()
    image = (image/scale)**0.5
    image *= scale
    
    img = Image.fromarray(255*(image/scale))
    rgb_img = img.convert('RGB')
    
    # save images:
    output_file = os.path.join(str_name, "img_sat{}.png".format(projection_index+1) )
    rgb_img.save(output_file)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    ax.set_axis_off()
    im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
    ax.set_title("{}".format(name)) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)           
    
plt.show()    


print('DONE forwared simulation')