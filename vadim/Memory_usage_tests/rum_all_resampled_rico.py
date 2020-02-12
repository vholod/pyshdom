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

M = 5
DO40 = False
DO80 = False
DO20 = False
DO10 = False
DO15 = False
DO30 = False
DO50 = True
DO100 = True
DO160 = False

# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO40):
    print( "--40--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr40/monochromatic'
    path = './tamar_cvpr_rico40x40x40.txt'
    cam_nx = 40
    cam_ny = 40
    mie_base_path = 'mie_tables/polydisperse/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO80):
    print( "--80--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr80/monochromatic'
    path = './tamar_cvpr_rico80x80x80.txt'
    cam_nx = 80
    cam_ny = 80
    mie_base_path = 'mie_tables/polydisperse_extanded/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)
            
                
            


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO20):
    print( "--20--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr20/monochromatic'
    path = './tamar_cvpr_rico20x20x20.txt'
    cam_nx = 20
    cam_ny = 20
    mie_base_path = 'mie_tables/polydisperse/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)
            
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO10):
    print( "--10--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr10/monochromatic'
    path = './tamar_cvpr_rico10x10x10.txt'
    cam_nx = 10
    cam_ny = 10
    mie_base_path = 'mie_tables/polydisperse/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)
            
              


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO160):
    print( "--160--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr160/monochromatic'
    path = './tamar_cvpr_rico160x160x160.txt'
    cam_nx = 160
    cam_ny = 160
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)
            
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO15):
    print( "--15--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr15/monochromatic'
    path = './tamar_cvpr_rico15x15x15.txt'
    cam_nx = 15
    cam_ny = 15
    mie_base_path = 'mie_tables/polydisperse_extanded/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO30):
    print( "--30--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr30/monochromatic'
    path = './tamar_cvpr_rico30x30x30.txt'
    cam_nx = 30
    cam_ny = 30
    mie_base_path = 'mie_tables/polydisperse_extanded/Water_672nm.scat'
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO50):
    print( "--50--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr50/monochromatic'
    path = './tamar_cvpr_rico50x50x50.txt'
    cam_nx = 50
    cam_ny = 50
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
if(DO100):
    print( "--100--")
    print( "njobs = 72")
    output_dir = 'experiments/help_tamar_cvpr100/monochromatic'
    path = './tamar_cvpr_rico100x100x100.txt'
    cam_nx = 100
    cam_ny = 100
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
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 72
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
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
            
            wavelength = 0.672
            # for each wavelength, the render script prepares its own rte_solver.
            generator = 'LesFile'
            solar_zenith = 180
            solar_azimuth = 0
            surface_albedo = 0.009999
            
            #camera parameters:
            cam_x = 0.4
            cam_y = 0.4
            cam_z = 1.73333
            
            lookat_x = 0.4
            lookat_y = 0.4
            lookat_z = 1.06667
            
            fov = 85.41877991472295
            
            n_jobs = 1
            
            # numerical parameters:
            num_mu = 16
            num_phi = 32
            split_accuracy = 0.01
            
            cmd = 'python -m memory_profiler '+ 'render_prespective_views.py'+\
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
                        ' --add_rayleigh'    
            
            Render_Perspective = subprocess.call( cmd, shell=True)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------