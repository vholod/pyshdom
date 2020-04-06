

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
air_path = './ancillary_data/AFGL_summer_mid_lat.txt'

print ("1. Generate Mie scattering tables for AirMSPI and more wavelengths")
# set parameteres for generate_mie_tables.py
start_reff = 4.0 # Starting effective radius [Micron]
end_reff = 25.0
num_reff = 50
start_veff = 0.01
end_veff = 0.2
num_veff = 50
radius_cutoff = 65.0 # The cutoff radius for the pdf averaging [Micron]
wavelength_list = [0.355, 0.38, 0.445, 0.47, 0.555, 0.66, 0.672, 0.865, 0.935, 1.5, 1.6]
"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_935nm.scat
"""
where_to_check_path = './mie_tables/polydisperse'
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
            wavelength_arg
    
    Generate_Mie_scat = subprocess.call( cmd, shell=True)
print('1. done')

print(20*'-')
print(20*'-')
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
IfRender_SINGLE_VOXEL = True
Render_flag = False
DOINVERSE = True
SEEIMAGES = False

if(IfRender_SINGLE_VOXEL):
    print("2. Render a single voxel atmosphere at 9 view angles, 20m resolution, 672nm")
    output_dir = 'experiments/single_voxel/monochromatic'
    log_name = 'vadim_single_voxel'
    # If the LWC is specified then the extinction is derived using
    #(lwc,reff,veff). If not the extinction needs to be directly
    #specified.
    lwc = 35e-3
    
    if(Render_flag):
        
        wavelength = 0.672
        #mie_base_path = 
        # for each wavelength, the render script prepares its own rte_solver.
        generator = 'SingleVoxel'
        
        
        domain_size = 1.0
        x_res = 0.02
        y_res = 0.02
        nx = 10
        ny = 10
        nz = 10
        # nx,ny,nz,domain_size are generator arguments
        azimuth_list = [90, 90, 90, 90, 0, -90, -90, -90, -90]
        azimuth_string = functools.reduce(operator.add,[str(j)+" " for j in azimuth_list]).rstrip()
        azimuth_arg = ' --azimuth '+azimuth_string
        
        zenith_list = [70.5, 60, 45.6, 26.1, 0.0, 26.1, 45.6, 60, 70.5]
        zenith_string = functools.reduce(operator.add,[str(j)+" " for j in zenith_list]).rstrip()
        zenith_arg = ' --zenith '+zenith_string
        
        extinction = 5.0 #' --extinction '+ str(extinction) +\
        reff = 10.0
        n_jobs = 20
        
        # numerical parameters:
        num_mu = 16
        num_phi = 32
        split_accuracy = 0.01
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'render_radiance_toa.py')+\
                ' ' + output_dir +\
                ' ' + str(wavelength) +\
                ' --generator '+ generator +\
                ' --domain_size '+ str(domain_size) +\
                ' --x_res '+ str(x_res) +' --y_res '+ str(y_res) +\
                ' --nx '+ str(nx) +' --ny '+ str(ny) +' --nz '+ str(nz) +\
                azimuth_arg +\
                zenith_arg +\
                ' --lwc '+ str(lwc) + ' --reff '+ str(reff) +\
                ' --n_jobs '+ str(n_jobs) +\
                ' --split_accuracy '+ str(split_accuracy) +\
                ' --num_mu '+ str(num_mu) +' --num_phi '+ str(num_phi) +\
                ' --add_rayleigh'
        
        
        Render_Orthographic = subprocess.call( cmd, shell=True)
    
    """
    Here the call of cmd command, rendered the images and
    saved measurements, medium and solver parameters.
    """
    print("2. done")
    print(20*'-')
    print(20*'-')
    
else: # render les:
    print("2. Render an LES cloud field (rico) at 9 view angles, 10m resolution, 672nm, with a rayleigh scattering atmosphere and parallelization")
    output_dir = 'experiments/LES_cloud_field_rico/monochromatic'
    log_name = 'rico32x37x26'
    
    if(Render_flag):
        
        wavelength = 0.672
        #mie_base_path = 
        # for each wavelength, the render script prepares its own rte_solver.
        generator = 'LesFile'
        path = '/home/vhold/pyshdom/synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
        
        domain_size = 1.0
        x_res = 0.02
        y_res = 0.02
        
        azimuth_list = [90, 90, 90, 90, 0, -90, -90, -90, -90]
        azimuth_string = functools.reduce(operator.add,[str(j)+" " for j in azimuth_list]).rstrip()
        azimuth_arg = ' --azimuth '+azimuth_string
        
        zenith_list = [70.5, 60, 45.6, 26.1, 0.0, 26.1, 45.6, 60, 70.5]
        zenith_string = functools.reduce(operator.add,[str(j)+" " for j in zenith_list]).rstrip()
        zenith_arg = ' --zenith '+zenith_string
        
        n_jobs = 72
        
        # numerical parameters:
        num_mu = 16
        num_phi = 32
        split_accuracy = 0.01
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'render_radiance_toa.py')+\
                ' ' + output_dir +\
                ' ' + str(wavelength) +\
                ' --generator '+ generator +\
                ' --path '+ path +\
                ' --x_res '+ str(x_res) +' --y_res '+ str(y_res) +\
                azimuth_arg +\
                zenith_arg +\
                ' --n_jobs '+ str(n_jobs) +\
                ' --split_accuracy '+ str(split_accuracy) +\
                ' --num_mu '+ str(num_mu) +' --num_phi '+ str(num_phi) +\
                ' --add_rayleigh'
        
        
        Render_Orthographic = subprocess.call( cmd, shell=True)
    
    """
    Here the call of cmd command, rendered the images and
    saved measurements, medium and solver parameters.
    """
    print("2. done")
    print(20*'-')
    print(20*'-')    
    
# -------------------------------------------------------
# -------------------------------------------------------
# ----------------------------------------------------------
"""
In this part try to set the radiance_threshold.
It is importent for the space curving (when option --use_forward_mask is off).
"""
DEBUG = True
if(DEBUG):
    
    # load the measurments to see the rendered images:
    medium, solver, measurements = shdom.load_forward_model(output_dir)    
    # Get optical medium ground-truth
    scatterer_name='cloud'
    ground_truth = medium.get_scatterer(scatterer_name)
    if isinstance(ground_truth, shdom.MicrophysicalScatterer):
        ground_truth = ground_truth.get_optical_scatterer(measurements.wavelength)

        
radiance_threshold = 0.0167
if(SEEIMAGES):
    
    measurements_path = output_dir
    measurements = shdom.load_forward_model_measurements(measurements_path)
    
    USED_CAMERA = measurements.camera
    RENDERED_IMAGES = measurements.images    
    THIS_MULTI_VIEW_SETUP = USED_CAMERA.projection
    # calculate images maximum:
    MAXI = 0
    for img in RENDERED_IMAGES:
        MAXI = max(MAXI,img.max())
        
    # show the renderings:
    RI = len(RENDERED_IMAGES)
    f, axarr = plt.subplots(2, RI, figsize=(20, 20))
    axarr = axarr.ravel()
    for ax, image in zip(axarr[:RI], RENDERED_IMAGES):

        ax.imshow(image,cmap='gray')
        ax.axis('off') 

    for ax, image in zip(axarr[RI:], RENDERED_IMAGES):
        image[image<=radiance_threshold] = 0
        ax.imshow(image,cmap='gray')
        ax.axis('off') 
    plt.show()    
    
if(DOINVERSE):
    all_known = False
    extinction_nomask = False
    Microphysical_find_lwc = False
    Microphysical_find_reff = True

    if(all_known):
        
        print("3. Estimate extinction with the ground truth phase function, grid and cloud mask")
        input_dir = output_dir
        # log_name defiend above
        
        n_jobs = 40
        extinction = 0.01
        log_name = log_name + '_all_known'
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
    
    if(extinction_nomask):
            
        print("4. Estimate extinction with the ground truth phase function, grid")
        input_dir = output_dir
        # log_name defiend above
        
        n_jobs = 40
        extinction = 0.01
        log_name = log_name + '_all_known-mask'
        # numerical parameters:
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_extinction_lbfgs.py')+\
                ' --input_dir ' + input_dir + \
                ' --add_rayleigh'+\
                ' --use_forward_grid'+\
                ' --use_forward_albedo'+\
                ' --use_forward_phase'+\
                ' --init Homogeneous'+\
                ' --radiance_threshold ' + str(radiance_threshold) +\
                ' --extinction ' + str(extinction) +\
                ' --log ' + log_name +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)
                
        Inverse_extinction = subprocess.call( cmd, shell=True)
        
        """
        Here the call of cmd command, rendered the images and
        saved measurements, medium and solver parameters.
        """
        print("4. done")
        print(20*'-')
        print(20*'-')        
        
# Estimate Micro-physical Properties        
    if(Microphysical_find_lwc):
            
        print("5. Estimate lwc with ground-truth effective radius and effective variance and use GT cloud mask")
        input_dir = output_dir
        # log_name defiend above
        
        n_jobs = 40
        lwc = 0.01
        
        log_name1 = log_name + '_Micro-physical-find_lwc-usemaks'
        # numerical parameters:
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                ' --input_dir ' + input_dir + \
                ' --add_rayleigh'+\
                ' --use_forward_grid'+\
                ' --use_forward_veff'+\
                ' --use_forward_reff'+\
                ' --use_forward_mask'+\
                ' --init Homogeneous'+\
                ' --lwc ' + str(lwc) +\
                ' --log ' + log_name1 +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)
                
        Inverse_extinction = subprocess.call( cmd, shell=True)

        print("5. done")
        print(20*'-')
        print(20*'-')     
        
        print("6. Estimate lwc with ground-truth effective radius and effective variance")
        input_dir = output_dir
        # log_name defiend above
        
        log_name2 = log_name + '_Micro-physical-find_lwc'
        # numerical parameters:
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                ' --input_dir ' + input_dir + \
                ' --add_rayleigh'+\
                ' --use_forward_grid'+\
                ' --use_forward_veff'+\
                ' --use_forward_reff'+\
                ' --init Homogeneous'+\
                ' --radiance_threshold ' + str(radiance_threshold) +\
                ' --lwc ' + str(lwc) +\
                ' --log ' + log_name2 +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)
                
        Inverse_extinction = subprocess.call( cmd, shell=True)

        print("6. done")
        print(20*'-')
        print(20*'-')         
        
    if(Microphysical_find_reff):
            
        print("7. Estimate lwc with ground-truth effective radius and effective variance and use GT cloud mask")
        input_dir = output_dir
        # log_name defiend above
        
        n_jobs = 40
        reff = 1
        
        log_name1 = log_name + '_Micro-physical-find_reff-usemaks'
        # numerical parameters:
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                ' --input_dir ' + input_dir + \
                ' --add_rayleigh'+\
                ' --use_forward_grid'+\
                ' --use_forward_veff'+\
                ' --use_forward_lwc'+\
                ' --use_forward_mask'+\
                ' --init Homogeneous'+\
                ' --reff ' + str(reff) +\
                ' --log ' + log_name1 +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)
                
        Inverse_extinction = subprocess.call( cmd, shell=True)

        print("7. done")
        print(20*'-')
        print(20*'-')     
        
        print("8. Estimate lwc with ground-truth effective radius and effective variance")
        input_dir = output_dir
        # log_name defiend above
        
        log_name2 = log_name + '_Micro-physical-find_reff'
        # numerical parameters:
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                ' --input_dir ' + input_dir + \
                ' --add_rayleigh'+\
                ' --use_forward_grid'+\
                ' --use_forward_veff'+\
                ' --use_forward_lwc'+\
                ' --init Homogeneous'+\
                ' --radiance_threshold ' + str(radiance_threshold) +\
                ' --reff ' + str(reff) +\
                ' --log ' + log_name2 +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)
                
        Inverse_extinction = subprocess.call( cmd, shell=True)

        print("8. done")
        print(20*'-')
        print(20*'-')             