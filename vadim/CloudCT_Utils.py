import os 
import sys
import scipy.io as sio
import numpy as np
import shdom 
import subprocess
import time
import glob
# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 


def CALC_MIE_TABLES(where_to_check_path = './mie_tables/polydisperse',wavelength_micron=None,options=None):
    where_to_check_path = os.path.join(where_to_check_path,'polydisperse')
    """
    Check if mie tables exist, if not creat them, if yes skip it is long process.
    table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
    Parameters
    ----------
    where_to_check_path: string, a path to chech (and or create) the mie table.
    
    wavelength_micron: float, the wavelength in microns
    """    
    
    # YOU MAY TUNE THE MIE TABLE PARAMETERS HERE:
    start_reff = options['start_reff'] # Starting effective radius [Micron]
    end_reff = options['end_reff']
    num_reff = options['num_reff']
    start_veff = options['start_veff']
    end_veff = options['end_veff']
    num_veff = options['num_veff']
    radius_cutoff = options['radius_cutoff'] # The cutoff radius for the pdf averaging [Micron]
    
    
    # --------------------------------------------------------------
    assert wavelength_micron is not None, "You must provied the wavelength"

        
    if(np.isscalar(wavelength_micron)):
        wavelength_list = [wavelength_micron]
    else:
        if(not isinstance(wavelength_micron, list)):
            wavelength_list = wavelength_micron.tolist()
        else:
            wavelength_list = wavelength_micron
    
    
    
    if(os.path.exists(where_to_check_path)):
        import glob        
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
                            
    
    if(not (wavelength_list_final==[])):
        
        wavelength_string = functools.reduce(operator.add,[str(j)+" " for j in wavelength_list_final]).rstrip()
        wavelength_arg = ' --wavelength '+wavelength_string
        print(wavelength_arg)
        """
        wavelength:
        Wavelengths [Micron] for which to compute the polarized mie table' \
        'The output file name is formated as: Water_<wavelength[nm]>nm.scat<pol>
        """
        
        #------------
        SCRIPTS_PATH = '../scripts'
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'generate_mie_tables.py')+\
                ' --start_reff '+ str(start_reff) +\
                ' --end_reff '+ str(end_reff) +\
                ' --num_reff '+ str(num_reff) +\
                ' --start_veff '+ str(start_veff) +\
                ' --end_veff '+ str(end_veff) +\
                ' --num_veff '+ str(num_veff) +\
                ' --radius_cutoff '+ str(radius_cutoff) +\
                ' --poly_dir ' + where_to_check_path +\
                ' --mono_dir ' + where_to_check_path.replace('polydisperse','monodisperse') +\
                wavelength_arg
        
        Generate_Mie_scat = subprocess.call( cmd, shell=True)
        
    print('Mie table is calculated.')

    print(20*'-')
    print(20*'-')
    
    mie_base_path = []
    for _wavelength_ in wavelength_list:
        mie_base_path.append(where_to_check_path + '/Water_{}nm.scat'. \
        format(int(1e3 * _wavelength_)))
        
    
    return mie_base_path




def SatSpeed(orbit = 100e3):
    """
    Determine the speed, (later, acceleration and orbital period) of the satellite.
    """
    Rsat = orbit # in meters.
    R = r_earth + Rsat #  the orbital radius
    # The orbital speed of the satellite using the following equation:
    # v = SQRT [ (G*M_earth ) / R ]
    V = np.sqrt((G*M_earth)/R) # units of [m/sec]
    return V    
# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS ABOVE----------------------
# -----------------------------------------------------------------
