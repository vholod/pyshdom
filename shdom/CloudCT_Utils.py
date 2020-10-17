import os 
import sys
import scipy.io as sio
import numpy as np
import shdom 
from shdom import float_round, core
import subprocess
import time
import glob
# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 
import re
import pickle

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
r_earth = 6371.0e3 # m 
M_earth = 5.98e24 # kg
G = 6.673e-11 # gravit in N m2/kg2
# h is the Planck's constant, c the speed of light,
h = 6.62607004e-34 #J*s is the Planck constant
c = 3.0e8 #m/s speed of litght
k = 1.38064852e-23 #J/K is the Boltzmann constant
# -------------------------------------------------------------------------------
# ----------------------FUNCTIONS------------------------------------------
# -------------------------------------------------------------------------------


def CALC_MIE_TABLES(where_to_check_path = './mie_tables',wavelength_micron=None,options=None,wavelength_averaging=False):
    base_path = where_to_check_path
    where_to_check_path = os.path.join(where_to_check_path,'polydisperse')
    """
    Check if mie tables exist, if not creat them, if yes skip it is long process.
    table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
    Parameters
    ----------
    where_to_check_path: string, a path to chech (and or create) the mie table.
    
    wavelength_micron: float or list of 2 float elements, the wavelength in microns.
    
    wavelength_averaging: bool,
        If it is False, Mie tables will be calculated for scattering properties of the central wavelength.
        The central wavelength is given wavelength_micron.
        
        If it is True, the input wavelength_micron must have a range of wavelengths [start,stop] in microns.
        The Mie tables will be calculated by averaging scattering properties over the wavelength band.
        In the options there must be a value of wavelength_resolution, it is the distance between two wavelength samples in the band.
        
    """    
    
    # YOU MAY TUNE THE MIE TABLE PARAMETERS HERE:
    #dr = options['dr']
    start_reff = options['start_reff'] # Starting effective radius [Micron]
    end_reff = options['end_reff']
    num_reff = options['num_reff'] # int((end_reff - start_reff)/dr) + 1 
    start_veff = options['start_veff']
    end_veff = options['end_veff']
    num_veff = options['num_veff']
    radius_cutoff = options['radius_cutoff'] # The cutoff radius for the pdf averaging [Micron]
    if(wavelength_averaging):
        assert 'wavelength_resolution' in options.keys(), 'When wavelength_averaging is True, you must provied the wavelength_resolution parameter!'
        wavelength_resolution = options['wavelength_resolution']
        
    
    # --------------------------------------------------------------
    assert wavelength_micron is not None, "You must provied the wavelength"

        
    if(np.isscalar(wavelength_micron)):
        wavelength_list = [wavelength_micron]
    else:
        if(not isinstance(wavelength_micron, list)):
            wavelength_list = wavelength_micron.tolist()
        else:
            wavelength_list = wavelength_micron
    
    
    if(wavelength_averaging):
        
        # Find central wavelength:
        # to do that, it is best to use the same method that shdom (original) uses.
        centeral_wavelength = core.get_center_wavelen(wavelength_list[0],wavelength_list[1])
        centeral_wavelength = float_round(centeral_wavelength)
        # It is close to (0.5*(wavelength_list[0]+wavelength_list[1]))
        
        if(not os.path.exists(where_to_check_path)):
            # safe creation of the directories:
            mono_directory = os.path.join(base_path,'monodisperse')
            if not os.path.exists(mono_directory):
                os.makedirs(mono_directory)
            
            poly_directory = os.path.join(base_path,'polydisperse')
            if not os.path.exists(poly_directory):
                os.makedirs(poly_directory)
                
        mie_tables_paths = sorted(glob.glob(where_to_check_path + '/averaged_Water_*.scat'))
        mie_tables_names = [os.path.split(i)[-1] for i in mie_tables_paths]
        # extract the wavelength:
        exist_wavelengths_list = [re.findall('averaged_Water_(\d*)nm.scat', i)[0] for i in mie_tables_names]# wavelength that already has mie table.
        exist_wavelengths_list = [int(i) for i in exist_wavelengths_list]# convert to integer 
        # the wavelength_list is in microne. the exist_wavelengths_list is in nm
        
        print('Check if wavelength {}um has already a table for the avareged case.'.format(centeral_wavelength))
        if(1e3*centeral_wavelength in exist_wavelengths_list):
            print('Does exist, skip its creation\n')
            print('But pay attention that you are using the same narrow band\n')
            mie_base_path = os.path.join(where_to_check_path, 'averaged_Water_{}nm.scat'.format(int(1e3*centeral_wavelength)))
            
        else:
            print('Does not exist, it will be created\n')
            # safe creation of the directories:
            mono_directory = os.path.join(base_path,'monodisperse')
            if not os.path.exists(mono_directory):
                os.makedirs(mono_directory)
            
            poly_directory = os.path.join(base_path,'polydisperse')
            if not os.path.exists(poly_directory):
                os.makedirs(poly_directory)
                
            #----------------------------------
            mie_mono = shdom.MieMonodisperse(particle_type='Water')
            mie_mono.set_wavelength_integration(wavelength_band=(wavelength_list[0], wavelength_list[1])
                                               ,wavelength_averaging=True,wavelength_resolution=wavelength_resolution)
            
            mie_mono.set_radius_integration(minimum_effective_radius=start_reff, max_integration_radius=radius_cutoff)
            mie_mono.compute_table()
            # save to mono dir
            output_path = os.path.join(mono_directory, 'averaged_Water_{}nm.scat'.format(int(1e3*centeral_wavelength)))
            mie_mono.write_table(output_path)
            
            # Compute a size distribution for several size distribution parameters
            size_distribution = shdom.SizeDistribution(type='gamma')
            size_distribution.set_parameters(reff=np.linspace(start_reff, end_reff, num_reff), veff=np.linspace(start_veff, end_veff, num_veff))
            size_distribution.compute_nd(radii=mie_mono.radii, particle_density=mie_mono.pardens)
            
            # Compute Polydisperse scattering for multiple size-distributions and save a polydisperse table.
            mie_poly = shdom.MiePolydisperse(mono_disperse=mie_mono, size_distribution=size_distribution)
            mie_poly.compute_table()
            output_path = os.path.join(poly_directory, 'averaged_Water_{}nm.scat'.format(int(1e3*centeral_wavelength)))
            mie_poly.write_table(output_path)                
             
            mie_base_path = output_path # it is the output of this function.   
            
    
    
    
        
    else: # treat the mie calculations for each wavelength separately.
    
        if(not os.path.exists(where_to_check_path)):
            # safe creation of the directories:
            mono_directory = os.path.join(base_path,'monodisperse')
            if not os.path.exists(mono_directory):
                os.makedirs(mono_directory)
            
            poly_directory = os.path.join(base_path,'polydisperse')
            if not os.path.exists(poly_directory):
                os.makedirs(poly_directory)
            
        
        mie_tables_paths = sorted(glob.glob(where_to_check_path + '/Water_*.scat'))
        mie_tables_names = [os.path.split(i)[-1] for i in mie_tables_paths]
        # extract the wavelength:
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
                            
        
        if(not (wavelength_list==[])):
            
            wavelength_string = functools.reduce(operator.add,[str(j)+" " for j in wavelength_list]).rstrip()
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

def save_CloudCT_measurments_and_forward_model(directory, medium, solver, measurements):
    """
    Save the forward model parameters for reconstruction.
    
    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved. 
        If the folder doesnt exist it will be created.
    medium: shdom.Medium object
        The atmospheric medium. This ground-truth medium will be used for comparisons.
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.SpaceMultiView_Measurements
        Contains the camera used and the measurements acquired. 
        
    Notes
    -----
    The ground-truth medium is later used for evaulation of the recovery.
    """  
    if not os.path.isdir(directory):
        os.makedirs(directory)  
    measurements.save(os.path.join(directory, 'cloudct_measurements'))
    
    medium.save(os.path.join(directory, 'ground_truth_medium'))
    solver.save_params(os.path.join(directory, 'solver_parameters'))   


def load_CloudCT_measurments_and_forward_model(directory):
    """
    Load the forward model parameters for reconstruction.
    
    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved. 
    
    Returns
    -------
    medium: shdom.Medium object
        The ground-truth atmospheric medium. 
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.Measurements
        Contains the sensor used to image the mediu and the radiance measurements. 
        
    Notes
    -----
    The ground-truth medium is used for evaulation of the recovery.
    """  
    # Load the ground truth medium for error analysis and ground-truth known phase and albedo
    medium_path = os.path.join(directory, 'ground_truth_medium')
    if os.path.exists(medium_path):
        medium = shdom.Medium()
        medium.load(path=medium_path)   
    else: 
        medium = None
        
    # Load RteSolver according to numerical and scene parameters
    solver_path = os.path.join(directory, 'solver_parameters')
    if np.array(medium.wavelength).size == 1:
        solver = shdom.RteSolver()
    else:
        solver = shdom.RteSolverArray()
    if os.path.exists(solver_path):
        solver.load_params(path=os.path.join(directory, 'solver_parameters'))   
    
    # Load the cloudct measurments:
    path = os.path.join(directory, 'cloudct_measurements')
    file = open(path, 'rb')
    data = file.read()
    CloudCT_measurments = pickle.loads(data)
    
    #CloudCT_measurments = shdom.Measurements()
    #measurements_path = os.path.join(directory, 'measurements')
    #assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
    #measurements.load(path=measurements_path)    
    
    
        
    return medium, solver, CloudCT_measurments


# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS ABOVE----------------------
# -----------------------------------------------------------------
