

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
Render_flag = True

if(IfRender_SINGLE_VOXEL):
    print("2. Render a single voxel atmosphere at 9 view angles, 20m resolution, 672nm")
    output_dir = 'experiments/single_voxel/monochromatic'
    if(Render_flag):
        
        wavelength = 0.672
        #mie_base_path = 
        # for each wavelength, the render script prepares its own rte_solver.
        generator = 'SingleVoxel'
        log_name = 'vadim_single_voxel'
        
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
        
        extinction = 5.0
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
                ' --extinction '+ str(extinction) +' --reff '+ str(reff) +\
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
    if(Render_flag):
        
        wavelength = 0.672
        #mie_base_path = 
        # for each wavelength, the render script prepares its own rte_solver.
        generator = 'LesFile'
        path = '/home/vhold/pyshdom/synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
        log_name = 'rico32x37x26'
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
print("3. Estimate extinction with the ground truth phase function, grid and cloud mask for precomputed LES measurements")
input_dir = output_dir
# log_name defiend above

n_jobs = 40
extinction = 0.01

air_path = '/home/vhold/pyshdom/vadim/ancillary_data/AFGL_summer_mid_lat.txt'
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

