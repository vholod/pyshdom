
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle
from shutil import copyfile
import time
import subprocess
import sys
import json
import glob
from contextlib import contextmanager

@contextmanager
def stdout_redirected(new_stdout):
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        
# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 

RENDER = True
SHOWIMAGES = True

M = 1
NJ1FLAG = False
#max_total_mb = 100000.0
max_total_mb = 10000.0
adapt_grid_factor = 10

# Only eliminating DELSOURCE will save a significant fraction of memory.
# when accelflag = true, _big_arrays = 3
# when accelflag = false, _big_arrays = 2
# what are the big_arrays in the init_memory in RteSolve?


# Define what is the gt_image and use it to calculate the image error.
# gt_image = ?

num_mu_list = [16]
num_phi_list = [32]

start_from_index = 5
end_index = start_from_index + 1

for num_sh_term_factor in [5]:
    for high_order_radiance in [True]: # it is faster to use it True
        for del_source in [False]: # it is faster to use it False
            # max bm:
            for max_total_mb in [320000.0]:
                # adapt_grid_factor 
                for adapt_grid_factor in [4]:

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
                    fx_list = fx_list[start_from_index:end_index]
                    fy_list = fy_list[start_from_index:end_index]
                    fz_list = fz_list[start_from_index:end_index]
                    origin_list = origin_list[start_from_index:end_index]
                    fov_list = fov_list[start_from_index:end_index]
                    lookat_list = lookat_list[start_from_index:end_index]


                    # without railight:
                    
                    
                    print( "without air")
                    # -------------------------------------------------------------
                    # -------------------------------------------------------------
                    # -------------------------------------------------------------
                    #start_from_index = 3
                    #end_index = 4

                    

                    for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
                        
                        for num_mu,num_phi in zip(num_mu_list,num_phi_list):

                            
                            
                            this_name = 'help_tamar_cvpr{}x{}x{}'.format(fx,fy,fz)
                            this_name = this_name + '_HOR' if high_order_radiance else this_name
                            this_name = this_name + '_DS' if del_source else this_name
                            this_name = this_name + '_ADF{}'.format(int(adapt_grid_factor))
                            this_name = this_name + '_MTB{}'.format(int(1e-3*max_total_mb))
                            this_name = this_name + '_DS' if del_source else this_name
                            this_name = this_name + '_NSTF{}E-1'.format(int(10*num_sh_term_factor))
                            this_name = this_name + '_MU{}PHI{}'.format(num_mu,num_phi)
    
    
                            output_dir = '../experiments/{}/monochromatic'.format(this_name)
                            
                            
                            #if not os.path.exists(output_dir):
                                #os.makedirs(output_dir)                                 
                            #sys.stdout = open(output_dir+'/summary.txt', 'w')
    
                            
                            path = 'mediums/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
    
                            print( "--{}x{}x{}--".format(fx,fy,fz))
                            print( "njobs = 72")
                            print( "max_total_mb = {}".format(max_total_mb))
                            print( "adapt_grid_factor = {}".format(adapt_grid_factor))
                            print( "del_source = {}".format(del_source))
                            print( "high_order_radiance = {}".format(high_order_radiance))
                            print( "num_sh_term_factor = {}".format(num_sh_term_factor))
                            print( "MU={}, PHI={}".format(num_mu,num_phi))
                            
                            cam_nx = fx
                            cam_ny = fy
    
                            mie_base_path = '../mie_tables/polydisperse/Water_672nm.scat'
                            for i in range(M):
                                print(20*'-')
                                print(20*'-')
                                print(i)
                                print(20*'-')
                                print(20*'-')
    
                                print("2. Render a Tamar's atmosphere, 672nm")
                                #if(not os.path.exists(output_dir)):
                                if(1):
    
                                    wavelength = 0.672
                                    # for each wavelength, the render script prepares its own rte_solver.
                                    generator = 'LesFile'
                                    solar_zenith = 180
                                    solar_azimuth = 0
                                    surface_albedo = 0.009999
    
                                    #camera parameters:
                                    cam_x = origin[0]
                                    cam_y = origin[1]
                                    cam_z = origin[2]
    
                                    lookat_x = lookat[0]
                                    lookat_y = lookat[1]
                                    lookat_z = lookat[2]
    
                                    # given in the loop
                                    # fov = 85.41877991472295
    
                                    n_jobs = 72
    
                                    # numerical parameters:
                                    # num_mu = 16
                                    # num_phi = 32
                                    split_accuracy = 0.01
    
                                    cmd = 'python -m memory_profiler '+ '../render_prespective_view.py'+\
                                        ' ' + output_dir +\
                                        ' ' + str(wavelength) +\
                                        ' --generator '+ generator +\
                                                ' --path '+ path +\
                                                ' --solar_zenith '+ str(solar_zenith) +' --solar_azimuth '+ str(solar_azimuth) +\
                                                ' --surface_albedo '+ str(surface_albedo) +\
                                                ' --cam_x '+ str(cam_x) +' --cam_y '+ str(cam_y) +' --cam_z '+ str(cam_z) +\
                                                ' --lookat_x '+ str(lookat_x) +' --lookat_y '+ str(lookat_y) +' --lookat_z '+ str(lookat_z) +\
                                                ' --fov '+ str(fov) +' --cam_nx '+ str(cam_nx) +' --cam_ny '+ str(cam_ny) +\
                                                            ' --n_jobs '+ str(n_jobs) +\
                                                            ' --split_accuracy '+ str(split_accuracy) +\
                                                ' --num_mu '+ str(num_mu) +' --num_phi '+ str(num_phi) +\
                                                ' --mie_base_path ' + mie_base_path +\
                                                ' --adapt_grid_factor '+ str(adapt_grid_factor) +\
                                                ' --num_sh_term_factor '+ str(num_sh_term_factor) +\
                                                ' --max_total_mb '+ str(max_total_mb)
    
                                    cmd = cmd + ' --del_source' if del_source else cmd # add the del_source            
                                    cmd = cmd + ' --high_order_radiance' if high_order_radiance else cmd
                                    #print(cmd)
                                    
                                    if(RENDER):
                                        Render_Perspective = subprocess.call( cmd, shell=True)
                                    
    
                            
                            
                                    if(SHOWIMAGES):
                                        
                                        if os.path.exists(output_dir):
                                            # load the measurments to see the rendered images:
                                            measurements = shdom.load_forward_model_measurements(output_dir)
                                            # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
                                            USED_CAMERA = measurements.camera
                                            RENCERED_IMAGES = measurements.images
                                            original_image = RENCERED_IMAGES[0]
                                            gamma_image = (original_image/np.max(original_image))**0.5
                                            
                                            f, ax = plt.subplots(1, 2, figsize=(20, 20))
                                            ax[0].imshow(original_image,cmap='gray')
                                            ax[0].invert_xaxis() 
                                            ax[0].invert_yaxis() 
                                            ax[0].axis('off')
                                            ax[0].set_title(this_name)
                                             
                                            # show the new image:
                                            original_image = RENCERED_IMAGES[1]
                                            gamma_image = (original_image/np.max(original_image))**0.5
                                            
                                            #f, ax = plt.subplots(1, 1, figsize=(20, 20))
                                            ax[1].imshow(original_image,cmap='gray')
                                            ax[1].invert_xaxis() 
                                            ax[1].invert_yaxis() 
                                            ax[1].axis('off')
                                            ax[1].set_title(this_name)                                    
                                        # calculate error:
                                        # epsilon = np.linalg.norm((original_image - gt_image), 2) / np.linalg.norm(gt_image,2)

                    # -------------------------------------------------------------
                    # -------------------------------------------------------------
                    # -------------------------------------------------------------
plt.show()