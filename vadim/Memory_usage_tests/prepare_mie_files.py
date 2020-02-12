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


SCRIPTS_PATH = '../scripts'
print ("1. Generate Mie scattering tables for AirMSPI and more wavelengths")
# set parameteres for generate_mie_tables.py
start_reff = 0.05 # Starting effective radius [Micron]
end_reff = 25.0
num_reff = 100
start_veff = 0.01
end_veff = 0.2
num_veff = 50
radius_cutoff = 65.0 # The cutoff radius for the pdf averaging [Micron]
wavelength_list = [0.672]
"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_935nm.scat
"""
where_to_check_path = './mie_tables/polydisperse_extanded_extra'
monodisperse = './mie_tables/monodisperse_extanded_extra'
if(os.path.exists(where_to_check_path)):
    mie_tables_paths = sorted(glob.glob(where_to_check_path + '/Water_*.scat'))
    mie_tables_names = [os.path.split(i)[-1] for i in mie_tables_paths]
    # extract the wavelength:
    import re
    exist_wavelengths_list = [re.findall('Water_(\d*)nm.scat', i)[0] for i in mie_tables_names]# wavelength that already has mie table.
    exist_wavelengths_list = [int(i) for i in exist_wavelengths_list]# convert to integer 
    # the wavelength_list is in microne. the exist_wavelengths_list is in nm
    wavelength_list_final = []
    for _wavelength_ in wavelength_list:
        print('Check if wavelength {}um has already a table.'.format(_wavelength_))
        if(1e3*_wavelength_ in exist_wavelengths_list):
            print('Does exist, skip its creation\n')
        else:
            print('Does not exist, it will be created\n')
            wavelength_list_final.append(_wavelength_)
            
    wavelength_list = wavelength_list_final
        

if(not (wavelength_list==[])):
    wavelength_string = functools.reduce(operator.add,[str(j)+" " for j in wavelength_list]).rstrip()
    wavelength_arg = ' --wavelength '+wavelength_string
    
    """
    wavelength:
    Wavelengths [Micron] for which to compute the polarized mie table' \
    'The output file name is formated as: Water_<wavelength[nm]>nm.scat<pol>
    """
    
    #------------
    
    cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'generate_mie_tables.py')+\
            ' --start_reff '+ str(start_reff) +\
            ' --end_reff '+ str(end_reff) +\
            ' --num_reff '+ str(num_reff) +\
            ' --start_veff '+ str(start_veff) +\
            ' --end_veff '+ str(end_veff) +\
            ' --num_veff '+ str(num_veff) +\
            ' --radius_cutoff '+ str(radius_cutoff) +\
            ' --poly_dir ' + where_to_check_path +\
            ' --mono_dir ' + monodisperse +\
            wavelength_arg
    
    Generate_Mie_scat = subprocess.call( cmd, shell=True)
print('1. done')

print(20*'-')
print(20*'-')
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
print( "njobs = 72")
output_dir = 'experiments/help_tamar_cvpr50/monochromatic'
path = './tamar_cvpr_rico50x50x50.txt'
cam_nx = 50
cam_ny = 50
mie_base_path = 'mie_tables/polydisperse_extanded/Water_672nm.scat'
M = 0
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
        
            
            
            
            
            
            
            
            
    """
    Here the call of cmd command, rendered the images and
    saved measurements, medium and solver parameters.
    """


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
ax[0].set_title('one view')

ax[1].imshow(gamma_image,cmap='gray')
ax[1].invert_xaxis() 
ax[1].invert_yaxis() 
ax[1].axis('off')
ax[1].set_title('one view gamma corrected')

plt.show()


print("2. done")
print(20*'-')
print(20*'-')

if(0):
    # -------------------------------------------------------
    # -------------------------------------------------------
    # ----------------------------------------------------------
    print("3. Estimate extinction with the ground truth phase function, grid and cloud mask for precomputed LES measurements")
    input_dir = '/home/vhold/pyshdom_v/vadim/experiments/single_voxel/monochromatic'
    
    n_jobs = 20
    extinction = 0.01
    log_name = 'vadim_single_voxel'
    air_path = '/home/vhold/pyshdom_v/vadim/ancillary_data/AFGL_summer_mid_lat.txt'
    # numerical parameters:
    cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_extinction_lbfgs.py')+\
            ' --input_dir ' + input_dir + \
            ' --add_rayleigh'+\
            ' --use_forward_grid'+\
            ' --use_forward_albedo'+\
            ' --use_forward_phase'+\
            ' --use_forward_mask'+\
            ' --init Homogeneous'+\
            ' --extinction ' + str(extinction) +\
            ' --log ' + log_name +\
            ' --air_path ' + air_path +\
            ' --n_jobs '+ str(n_jobs)
            
    Inverse_extinction = subprocess.call( cmd, shell=True)
    
    """
    Here the call of cmd command, rendered the images and
    saved measurements, medium and solver parameters.
    """
    print("3. done")
    print(20*'-')
    print(20*'-')
    
