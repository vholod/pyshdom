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

M = 1
NJ1FLAG = False
#max_total_mb = 100000.0
max_total_mb = 10000.0
adapt_grid_factor = 10

# Only eliminating DELSOURCE will save a significant fraction of memory.
# when accelflag = true, _big_arrays = 3
# when accelflag = false, _big_arrays = 2
# what are the big_arrays in the init_memory in RteSolve?
del_source = False

for high_order_radiance in [True, False]:
    for del_source in [False]:
        # max bm:
        for max_total_mb in [320000.0]:
            # adapt_grid_factor 
            for adapt_grid_factor in [10.0]:
                
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
                for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
                    
                    loop_index = fx_list.index(fx)
                    if(start_from_index>loop_index):
                        continue
                        
                    print( "--{}x{}x{}--".format(fx,fy,fz))
                    print( "njobs = 72")
                    print( "max_total_mb = {}".format(max_total_mb))
                    print( "adapt_grid_factor = {}".format(adapt_grid_factor))
                    print( "del_source = {}".format(del_source))
                    print( "high_order_radiance = {}".format(high_order_radiance))
                    
                    output_dir = '../experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
                    path = 'mediums/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
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
                            num_mu = 16
                            num_phi = 32
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
                                        ' --adapt_grid_factor '+ str(adapt_grid_factor) + ' --add_rayleigh ' + ' --max_total_mb '+ str(max_total_mb)   
                            
                            cmd = cmd + ' --del_source' if del_source else cmd # add the del_source
                            cmd = cmd + ' --high_order_radiance' if high_order_radiance else cmd
                            Render_Perspective = subprocess.call( cmd, shell=True)
                        
                        
                    # -------------------------njobs=1----------------------
                    if(NJ1FLAG):
                        print( "njobs = 1")
                        print( "max_total_mb = {}".format(max_total_mb))
                        print( "adapt_grid_factor = {}".format(adapt_grid_factor))
                        print( "del_source = {}".format(del_source))
                        print( "high_order_radiance = {}".format(high_order_radiance))
                        
                        for i in range(M):
                            print(20*'-')
                            print(20*'-')
                            print(i)
                            print(20*'-')
                            print(20*'-')
                            
                            print("2. Render a Tamar's atmosphere, 672nm")
                            #if(not os.path.exists(output_dir)):
                            
                                
                                
                            n_jobs = 1
                            
                           
                            
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
                                        ' --add_rayleigh' + ' --max_total_mb '+ str(max_total_mb)    
                            
                            cmd = cmd + ' --del_source' if del_source else cmd # add the del_source
                            cmd = cmd + ' --high_order_radiance' if high_order_radiance else cmd
                            Render_Perspective = subprocess.call( cmd, shell=True)
                    
                    
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                
                
                
                # without railight:
                print( "without air")
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                start_from_index = 2
                for fx,fy,fz,origin,fov,lookat in zip(fx_list,fy_list,fz_list,origin_list,fov_list,lookat_list):
                    
                    loop_index = fx_list.index(fx)
                    if(start_from_index>loop_index):
                        continue
                        
                    print( "--{}x{}x{}--".format(fx,fy,fz))
                    print( "njobs = 72")
                    print( "max_total_mb = {}".format(max_total_mb))
                    print( "adapt_grid_factor = {}".format(adapt_grid_factor))
                    print( "del_source = {}".format(del_source))
                    print( "high_order_radiance = {}".format(high_order_radiance))
                    
                    output_dir = '../experiments/help_tamar_cvpr{}x{}x{}/monochromatic'.format(fx,fy,fz)
                    path = 'mediums/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
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
                            num_mu = 16
                            num_phi = 32
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
                                        ' --max_total_mb '+ str(max_total_mb)
                            
                            cmd = cmd + ' --del_source' if del_source else cmd # add the del_source            
                            cmd = cmd + ' --high_order_radiance' if high_order_radiance else cmd
                            Render_Perspective = subprocess.call( cmd, shell=True)
                        
                        
                    # -------------------------njobs=1----------------------
                    if(NJ1FLAG):
                        print( "njobs = 1")
                        print( "max_total_mb = {}".format(max_total_mb))
                        print( "adapt_grid_factor = {}".format(adapt_grid_factor))
                        print( "del_source = {}".format(del_source))
                        print( "high_order_radiance = {}".format(high_order_radiance))
                    
                    
                        for i in range(M):
                            print(20*'-')
                            print(20*'-')
                            print(i)
                            print(20*'-')
                            print(20*'-')
                            
                            print("2. Render a Tamar's atmosphere, 672nm")
                            #if(not os.path.exists(output_dir)):
                            
                                
                                
                            n_jobs = 1
                            
                           
                            
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
                                        ' --max_total_mb '+ str(max_total_mb)
                            
                            cmd = cmd + ' --del_source' if del_source else cmd # add the del_source
                            cmd = cmd + ' --high_order_radiance' if high_order_radiance else cmd
                            Render_Perspective = subprocess.call( cmd, shell=True)
                
                
                # -------------------------------------------------------------
                # -------------------------------------------------------------
                # -------------------------------------------------------------
