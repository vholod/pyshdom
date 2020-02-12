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

M = 2
max_total_mb = 100000.0
#max_total_mb = 10000.0

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
start_from_index = 0
for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
    
    loop_index = fx_list.index(fx)
    if(start_from_index>loop_index):
        continue
        
    print( "--{}x{}x{}--".format(fx,fy,fz))
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
    path = '/home/vhold/pyshdom/notebooks/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
    cam_nx = fx
    cam_ny = fy
    
    mie_base_path = 'mie_tables/polydisperse_extanded_extra/Water_672nm.scat'
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
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_view.py'+\
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
                        ' --add_rayleigh ' + ' --max_total_mb '+ str(max_total_mb)   
            
            Render_Perspective = subprocess.call( cmd, shell=True)
        
        
    # -------------------------njobs=1----------------------
    print( "njobs = 1")
    for i in range(M):
        print(20*'-')
        print(20*'-')
        print(i)
        print(20*'-')
        print(20*'-')
        
        print("2. Render a Tamar's atmosphere, 672nm")
        #if(not os.path.exists(output_dir)):
        if(1):
            
            
            n_jobs = 1
            
           
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_view.py'+\
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
                        ' --add_rayleigh' + ' --max_total_mb '+ str(max_total_mb)    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------



# without railight:
print( "without air")
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
start_from_index = 0
for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
    
    loop_index = fx_list.index(fx)
    if(start_from_index>loop_index):
        continue
        
    print( "--{}x{}x{}--".format(fx,fy,fz))
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
    path = '/home/vhold/pyshdom/notebooks/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
    cam_nx = fx
    cam_ny = fy
    
    mie_base_path = 'mie_tables/polydisperse_extanded_extra/Water_672nm.scat'
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
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_view.py'+\
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
                        ' --max_total_mb '+ str(max_total_mb)
            
            Render_Perspective = subprocess.call( cmd, shell=True)
        
        
    # -------------------------njobs=1----------------------
    print( "njobs = 1")
    for i in range(M):
        print(20*'-')
        print(20*'-')
        print(i)
        print(20*'-')
        print(20*'-')
        
        print("2. Render a Tamar's atmosphere, 672nm")
        #if(not os.path.exists(output_dir)):
        if(1):
            
            
            n_jobs = 1
            
           
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_view.py'+\
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
                        ' --max_total_mb '+ str(max_total_mb)
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
