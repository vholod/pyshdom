import os 
import sys
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom 
from shdom import CloudCT_setup, float_round, core
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
from shdom.CloudCT_Utils import *

# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# mia table parameters:
start_reff = 4
end_reff = 25.0
start_veff = 0.01
end_veff = 0.4
radius_cutoff = 65.0
mie_options = {
    'start_reff': start_reff,# Starting effective radius [Micron]
    'end_reff': end_reff,
    'num_reff': 100,
    'start_veff': start_veff,
    'end_veff': end_veff,
    'num_veff': 117,
    'radius_cutoff': radius_cutoff, # The cutoff radius for the pdf averaging [Micron]
    'wavelength_resolution':0.001 # it is the delta in the wavelengths (microns) for the optical parameters avaraging inside shdom.
}



# visualization params:
VISSETUP = False
scale = 500
axisWidth = 0.02
axisLenght = 5000  


# debug params:
CENCEL_AIR = False # just for debuging, in rutine it must be false.

# Similation flags
DOFORWARD = True
DOINVERSE = True
USE_VIS = True # if to use vis in the simulation
USE_SWIR = False # if to use vis in the simulation


# numerical parameters
n_jobs = 28



# orbital setup parameters:
Rsat = 500 # km
GSD = 0.02 # in km, it is the ground spatial resolution.
SATS_NUMBER_SETUP = 10 # satellites number to build the setup, for the inverse, we can use less satellites.
# to consider - SATS_NUMBER_INVERSE = SATS_NUMBER_SETUP#10 # how much satelliets will be used for the inverse.

# where different imagers are located:
vis_imager_config = SATS_NUMBER_SETUP*[True]
#swir_imager_config = SATS_NUMBER_SETUP*[True]
swir_imager_config = SATS_NUMBER_SETUP*[False]
swir_imager_config[5] = True
swir_imager_config[4] = True

# solar irradiance parameters:
sun_azimuth = 45
sun_zenith = 150
#azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
#zenith: Solar beam zenith angle in range (90,180] 

# ----------------------------------------------------------
"""
Here load the imagers, the imagers dictates the spectral bands of the rte solver and rendering.
Since the spectrum in this script referes to narrow bands, the nerrow bands will be treated in the following maner:
We will use the wavelength averaging of shdom. It averages scattering properties over the wavelength band.
Be carfule, the wavelengths in Imager methods are in nm. Pyshdom the wavelength are usualy in microns.

THe imagers aslo dictate the ground spatial resolution (GSD).
"""

USE_SIMPLE_IMAGER = False # if it's true, the imager is simple and not loaded from the imagers data set.
if(not USE_SIMPLE_IMAGER):
    
    # load Imager at VIS:
    vis_imager = shdom.Imager.ImportConfig(file_name = '../notebooks/Gecko_config.json')
    # load Imager at SWIR:
    swir_imager = shdom.Imager.ImportConfig(file_name = '../notebooks/Hypothetic_SWIR_camera_config.json')
else:
    # the imager is simple it will be defined here:
    simple_sensor = shdom.SensorFPA(PIXEL_SIZE = 1,CHeight = 100, CWidth = 100)
    simple_lens = shdom.LensSimple(FOCAL_LENGTH = 1 , DIAMETER = 1)
    # shdom.LensSimple means that the lens model is simlpe and without MTF considerations but still with focal and diameter.
    
    
    swir_imager = shdom.Imager(sensor=simple_sensor,lens=simple_lens)
    vis_imager = shdom.Imager(sensor=simple_sensor,lens=simple_lens)
    # you must define the spectrum for each defined imager:
    swir_imager.set_scene_spectrum_in_microns([1.6,1.6])
    vis_imager.set_scene_spectrum_in_microns([0.672,0.672])
    
    
# set the nadir view altitude:
vis_imager.set_Imager_altitude(H=Rsat)
swir_imager.set_Imager_altitude(H=Rsat)

if(not USE_SIMPLE_IMAGER):
    # the following parametere refers only to nadir view.
    vis_pixel_footprint, _ = vis_imager.get_footprints_at_nadir()
    swir_pixel_footprint, _ = swir_imager.get_footprints_at_nadir()
    
    vis_pixel_footprint = float_round(vis_pixel_footprint)
    swir_pixel_footprint = float_round(swir_pixel_footprint)
    
else:
    # set required pixel footprint:
    vis_pixel_footprint = 0.02 # km  
    swir_pixel_footprint = 0.02 # km
    vis_imager.set_pixel_footprint(vis_pixel_footprint)
    swir_imager.set_pixel_footprint(swir_pixel_footprint)
     
# play with temperatures:
temp  = 15 # celsius
if(not USE_SIMPLE_IMAGER):
    # if we want to use the simple imager, the temperature does not make any differences.
    swir_imager.change_temperature(temp)
    vis_imager.change_temperature(temp)


# update solar irradince with the solar zenith angel:
vis_imager.update_solar_angle(sun_zenith)
swir_imager.update_solar_angle(sun_zenith)
# TODO -  use here pysat   or pyEpham package to predict the orbital position of the nadir view sattelite.
# It will be used to set the solar zenith.

# calculate mie tables:
vis_wavelegth_range_nm = vis_imager.scene_spectrum # in nm
swir_wavelegth_range_nm = swir_imager.scene_spectrum # in nm
# convert nm to microns: It is must
vis_wavelegth_range = [float_round(1e-3*w) for w in vis_wavelegth_range_nm]
swir_wavelegth_range = [float_round(1e-3*w) for w in swir_wavelegth_range_nm]

# central wavelengths.
vis_centeral_wavelength = vis_imager.centeral_wavelength_in_microns
swir_centeral_wavelength = swir_imager.centeral_wavelength_in_microns
wavelengths_micron = [vis_centeral_wavelength, swir_centeral_wavelength]
# wavelengths_micron will hold the vis swir wavelengths. It will be convinient to use the 
# wavelengths_micron in some loops.

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/avareged_Water_<1000*wavelength_micron>nm.scat
"""
MieTablesPath = os.path.abspath("./mie_tables")
if(vis_wavelegth_range[0] == vis_wavelegth_range[1]):

    vis_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                    vis_wavelegth_range[0],mie_options,wavelength_averaging=False)
else:
    
    vis_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                    vis_wavelegth_range,mie_options,wavelength_averaging=True)

if(swir_wavelegth_range[0] == swir_wavelegth_range[1]):
    
    swir_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                    swir_wavelegth_range[0],mie_options,wavelength_averaging=False)
    
else:
    
    swir_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                    swir_wavelegth_range,mie_options,wavelength_averaging=True)

# where to save the forward outputs:
forward_dir = './experiments/VIS_SWIR_NARROW_BANDS_VIS_{}-{}nm_active_sats_{}_GSD_{}m_and_SWIR__{}-{}nm_active_sats_{}_GSD_{}m_LES_cloud_field_rico_LES_cloud_field_rico'.\
    format(int(1e3 *vis_wavelegth_range[0]),int(1e3 *vis_wavelegth_range[1]),SATS_NUMBER_SETUP,int(1e3*vis_pixel_footprint),int(1e3 *swir_wavelegth_range[0]),int(1e3 *swir_wavelegth_range[1]),SATS_NUMBER_SETUP,int(1e3*swir_pixel_footprint))

# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name_base = 'active_sats_{}_easiest_rico32x37x26'.format(SATS_NUMBER_SETUP)
# Write intermediate TensorBoardX results into log_name.
# The provided string is added as a comment to the specific run.

# --------------------------------------------------------


# ---------------------------------------------------------------
# -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
# ---------------------------------------------------------------

CloudFieldFile = '../synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
if(CENCEL_AIR):
    AirFieldFile = None # here the atmosphere will note condsider any air.
    
else:
    AirFieldFile = './ancillary_data/AFGL_summer_mid_lat.txt' # Path to csv file which contains temperature measurements


if(not USE_SIMPLE_IMAGER): 
    # If we do not use simple imager we probably use imager with a band so:
    wavelength_averaging=True
else:
    wavelength_averaging = False
    
    
atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile,AirFieldFile,
                             MieTablesPath,[vis_centeral_wavelength, swir_centeral_wavelength],wavelength_averaging=wavelength_averaging)
droplets = atmosphere.get_scatterer('cloud')

# -----------------------------------------------
# ---------Set relevant camera parameters. ------
# ---For that we need some mediume sizes --------
# -----------------------------------------------
droplets_grid = droplets.grid
dx = droplets_grid.dx
dy = droplets_grid.dy

nz = droplets_grid.nz
nx = droplets_grid.nx
ny = droplets_grid.ny

Lx = droplets_grid.bounding_box.xmax - droplets_grid.bounding_box.xmin
Ly = droplets_grid.bounding_box.ymax - droplets_grid.bounding_box.ymin
Lz = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
L = max(Lx,Ly)

Lz_droplets = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
dz = Lz_droplets/(nz-1)

#USED FOV, RESOLUTION and SAT_LOOKATS:
# cny x cnx is the camera resolution in pixels
fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
vis_cny = int(np.floor(L/vis_pixel_footprint))
vis_cnx = int(np.floor(L/vis_pixel_footprint))

swir_cny = int(np.floor(L/swir_pixel_footprint))
swir_cnx = int(np.floor(L/swir_pixel_footprint))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]

# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if(IFTUNE_CAM):
    L = 1.5*L
    fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
    vis_cny = int(np.floor(L/vis_pixel_footprint))
    vis_cnx = int(np.floor(L/vis_pixel_footprint))
    
    swir_cny = int(np.floor(L/swir_pixel_footprint))
    swir_cnx = int(np.floor(L/swir_pixel_footprint))       



# Update the resolution of each Imager with respect to new pixels number [nx,ny].
# In addition we update Imager's FOV.
vis_imager.update_sensor_size_with_number_of_pixels(vis_cnx, vis_cny)
swir_imager.update_sensor_size_with_number_of_pixels(swir_cnx, swir_cny)

    
# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variavle LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
if(IFTUNE_CAM):
    LOOKAT[2] = 0.68*nx*dz # tuning. if IFTUNE_CAM = False, just lookat the bottom
        
SAT_LOOKATS = np.array(SATS_NUMBER_SETUP*LOOKAT).reshape(-1,3)# currently, all satellites lookat the same point.
    
print(20*"-")
print(20*"-")
print(20*"-")

print("CAMERA intrinsics summary")
print("vis: fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov,vis_cnx,vis_cny))
print("swir: fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov,swir_cnx,swir_cny))

print(20*"-")
print(20*"-")
print(20*"-")

print("Medium summary")
print("nx = {}, ny = {},nz ={}".format(nx,ny,nz))
print("dx = {}, dy = {},dz ={}".format(dx,dy,dz))
print("Lx = {}, Ly = {},Lz ={}".format(Lx,Ly,Lz))
x_min = droplets_grid.bounding_box.xmin
x_max = droplets_grid.bounding_box.xmax

y_min = droplets_grid.bounding_box.ymin
y_max = droplets_grid.bounding_box.ymax

z_min = droplets_grid.bounding_box.zmin
z_max = droplets_grid.bounding_box.zmax 
print("xmin = {}, ymin = {},zmin ={}".format(x_min,y_min,z_min))
print("xmax = {}, ymax = {},zmax ={}".format(x_max,y_max,z_max))

print(20*"-")
print(20*"-")
print(20*"-")


if(DOFORWARD):   
    # ---------------------------------------------------------------
    # ---------------CREATE THE SETUP----------------------------
    # ---------------------------------------------------------------
    """
    The forward simulation will run with unity flux in the input.
    Imager.L_TOA is a scale that we need to apply on the output images.
    """
      
    # create CloudCT setups:
    vis_CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(\
        SATS_NUMBER = SATS_NUMBER_SETUP, ORBIT_ALTITUDE = Rsat, \
        SAT_LOOKATS = SAT_LOOKATS, \
        Imager_config = vis_imager_config ,imager = vis_imager, VISSETUP = VISSETUP)
    
    swir_CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(\
        SATS_NUMBER = SATS_NUMBER_SETUP, ORBIT_ALTITUDE = Rsat, \
        SAT_LOOKATS = SAT_LOOKATS, \
        Imager_config = swir_imager_config ,imager = swir_imager, VISSETUP = VISSETUP)    
        
    # ----------------------------------------------------------
    # ---------numerical & scene Parameters---------------------
    # ---------for RTE solver and initializtion of the solver---
    # ----------------------------------------------------------
    solar_fluxes = [1,1] # unity flux, IT IS IMPORTANT HERE!
    split_accuracies = [0.1, 0.1]
    surface_albedos = [0.05, 0.05] # later we must update it since it depends on the wavelegth.
    # split_accuracy of 0.1 gives nice results, For the rico cloud i didn't see that decreasing the split accuracy improves the rec.
    # so fo the rico loud Let's use split_accuracy = 0.1.
    num_mu = 16
    num_phi = 32
    # note: whae num_mu, num_phi are 16,32, the retrievals look better than with 8,16.
    adapt_grid_factor = 5
    solution_accuracy = 0.0001
    max_total_mb = 100000.0
    # Generate a solver array for a multispectral solution.
    # it is greate that we can use the parallel solution of all solvers.
    # -----IMPORTANT NOTE---------
    # Rigth now the /numerical parameter of vis and swir are the same:
    
    rte_solvers = shdom.RteSolverArray()
    
    # TODO - what if the imagers have the same central wavelenghts?
    for wavelength,split_accuracy,solar_flux,surface_albedo in \
        zip(wavelengths_micron,split_accuracies,solar_fluxes,surface_albedos): 
        # iter 0 of wavelength is for vis
        # iter 1 of wavelength is for swir
        
        numerical_params = shdom.NumericalParameters(num_mu_bins=num_mu,num_phi_bins=num_phi,
                                                             split_accuracy=split_accuracy,max_total_mb=max_total_mb)
    
        scene_params = shdom.SceneParameters(
                    wavelength=wavelength,
                    surface=shdom.LambertianSurface(albedo=surface_albedo),
                    source=shdom.SolarSource(azimuth=sun_azimuth, zenith=sun_zenith,flux=solar_flux)
                ) 
        # ---------initilize an RteSolver object--------- 
        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)
        rte_solvers.add_solver(rte_solver) 

    # -----------------------------------------------
    # ---------RTE SOLVE ----------------------------
    # -----------------------------------------------
    maxiter = 150
    rte_solvers.solve(maxiter=maxiter)
        
    # -----------------------------------------------
    # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
    # -----------------------------------------------
    
    """
    Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
    The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
    """
    #n_jobs = 1 # when debugging
    
    setup_of_views_list  = [vis_CloudCT_VIEWS, swir_CloudCT_VIEWS]
    # the order of the bands is importent here.
    CloudCT_measurments = CloudCT_setup.SpaceMultiView_Measurements(\
        setup_of_views_list)
    
    if(not USE_SIMPLE_IMAGER): 
        # If we do not use simple imager we probably use imager such that noise can be applied:    
        CloudCT_measurments.simulate_measurements(n_jobs = n_jobs,rte_solvers=rte_solvers,IF_REDUCE_EXPOSURE=True,IF_SCALE_IDEALLY=False,IF_APPLY_NOISE=True)
    else:
        # If we do not use realistic imager, we MUST use IF_SCALE_IDEALLY=True so the images will have values in maximum range.
        CloudCT_measurments.simulate_measurements(n_jobs = n_jobs,rte_solvers=rte_solvers,IF_REDUCE_EXPOSURE=False,IF_SCALE_IDEALLY=True,IF_APPLY_NOISE=False)
    
    # The simulate_measurements() simulate images in gray levels.
    # Now we need to save that images and be able to load that images in the inverse pipline/
    
    
    # See the simulated images:
    SEE_IMAGES = True
    if(SEE_IMAGES):    
        CloudCT_measurments.show_measurments()
        plt.show()
        
    medium = atmosphere
    shdom.save_CloudCT_measurments_and_forward_model(directory = forward_dir, medium = medium, solver = rte_solvers , measurements = CloudCT_measurments)
    
    print('DONE forwared simulation')
    
    

# -----------------------------------------------
# ---------SOLVE INVERSE ------------------------
# -----------------------------------------------
"""
Solve inverse problem:
"""
if(DOINVERSE):
    
    # load the cloudct measurments to see the rendered images:
    medium, solver, CloudCT_measurments = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
    # A CloudCT Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    
    # See the simulated images:
    SEE_IMAGES = False
    if(SEE_IMAGES):    
        CloudCT_measurments.show_measurments()
        plt.show()
        
    # ---------what to optimize----------------------------
        
    # -----------------------------------------------
    # ---------SOLVE for lwc and reff  ------------------
    # -----------------------------------------------    
    """
    Estimate lwc with (what is known?):
    1. ground truth phase function (use_forward_phase, with mie_base_path)
    2. grid (use_forward_grid)
    3. rayleigh scattering (add_rayleigh)
    4. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

    """
    log_name = "rec_all_"+log_name_base
    # -----------------------------------------------
    # ---------------- inverse parameters:-----------
    # -----------------------------------------------
    
    stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
            
    # Initialization and Medium parameters:
    lwc = 0.01 # init lwc of the generator
    reff = 12 # init reff of the generator
    #veff = 0.15 # init veff of the generator
    
    # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
    init = 'Homogeneous'
    # The mie_base_path is defined at the begining of this script.
    INIT_USE = ' --init '+ init
    add_rayleigh = False if CENCEL_AIR else True
    use_forward_mask = True#False#True
    use_forward_grid = True
    if_save_gt_and_carver_masks = True
    if_save_final3d = True
    cloudct_use = True
    
    GT_USE = ''
    GT_USE = GT_USE + ' --cloudct_use' if cloudct_use else GT_USE # -----------------                           
    GT_USE = GT_USE + ' --use_forward_veff --lwc_scaling 15 --reff_scaling 0.01'# -----------------                       
    GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
    GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
    GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE               
    GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
    GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
    
    # optimization:
    globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
    maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
    maxls = 30 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
    disp = True # Display optimization progression.
    
    gtol = 1e-16 # Stop criteria for the maximum projected gradient.
    # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    ftol = 1e-16 # Stop criteria for the relative change in loss function.
    # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    loss_type = 'l2'
    # Different loss functions for optimization. Currently only l2 is supported.
    
    #-------------------------------------------------
    # ---- start working with the above parameters:---
    #-------------------------------------------------
    # ' --veff ' + str(veff) +\
    OTHER_PARAMS = ' --input_dir ' + forward_dir + \
        ' --lwc ' + str(lwc) +\
        ' --reff ' + str(reff) +\
        ' --log ' + log_name +\
        ' --n_jobs '+ str(n_jobs)+\
        ' --loss_type '+ str(loss_type)+\
        ' --maxls '+ str(maxls)+\
        ' --maxiter '+ str(maxiter)
        #' --radiance_threshold '+ str(radiance_threshold)
    
    OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
    OTHER_PARAMS = OTHER_PARAMS + ' --air_path ' + AirFieldFile if (CENCEL_AIR == False) else OTHER_PARAMS
    # We have: ground_truth, rte_solver, measurements.
    SCRIPTS_PATH = '../scripts'
    cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
        OTHER_PARAMS + GT_USE + INIT_USE
    
    Optimaize1 = subprocess.call( cmd, shell=True)         
    

print('done')

if(0):
    
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------    

    # that the time to show the results in 3D visualization:
    VIS_RESULTS3D = False
    if(VIS_RESULTS3D):    
        """
        The forward_dir id a folder that containes:
        medium, solver, measurements.
        They loaded before. To see the final state, the medium is not 
        enough, the medium_estimator is needed.
        load the measurments to see the rendered images:
        """
        
        
        # what state to load? I prefere the last one!
        logs_dir = os.path.join(forward_dir,'logs')
        logs_prefix = os.path.join(logs_dir,log_name)
        logs_files = glob.glob(logs_prefix + '-*')
        
        times = [i.split('{}-'.format(int(1e3*wavelength_micron)))[-1] for i in logs_files]
        # sort the times to find the last one.
        timestamp = [time.mktime(time.strptime(i,"%d-%b-%Y-%H:%M:%S")) for i in times]
        # time.mktime(t) This is the inverse function of localtime()
        timestamp.sort()
        timestamp = [time.strftime("%d-%b-%Y-%H:%M:%S",time.localtime(i)) for i in timestamp]
        # now, the timestamp are sorted, and I want the last stamp to visualize.
        connector = '{}-'.format(int(1e3*wavelength_micron))
        log2load = logs_prefix+'-'+timestamp[-1]
        # print here the Final results files:
        Final_results_3Dfiles = glob.glob(log2load + '/FINAL_3D_*.mat')
        print("{} files with the results in 3D were created:".format(len(Final_results_3Dfiles)))
        for _file in Final_results_3Dfiles:
            
            print(_file)
            
        # ---------------------------------------------------------
        
        # Don't want to use it now, state2load = os.path.join(log2load,'final_state.ckpt')
        
    
        print("use tensorboard --logdir {} --bind_all".format(log2load))
    
print("done")


# This distribution can be excellently approximated by a Gaussian
#distribution having this expectation and variance, for nphoto
#electr > 10, which is
#typically the case for cameras.