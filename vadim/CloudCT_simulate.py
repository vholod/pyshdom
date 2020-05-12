import os 
import sys
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom 
from shdom import CloudCT_setup
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
from CloudCT_Utils import *

# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# mia table parameters:
start_reff = 1
end_reff = 25.0
start_veff = 0.05
end_veff = 0.4
radius_cutoff = 65.0
mie_options = {
    'start_reff': start_reff,# Starting effective radius [Micron]
    'end_reff': end_reff,
    'num_reff': int((end_reff-start_reff)/0.25 + 1),
    'start_veff': start_veff,
    'end_veff': end_veff,
    'num_veff': int((end_veff-start_veff)/0.003 + 1),
    'radius_cutoff': radius_cutoff # The cutoff radius for the pdf averaging [Micron]
}

# visualization params:
VISSETUP = False
scale = 500
axisWidth = 0.02
axisLenght = 5000  

# sat number:
#SATS_NUMBER = 10
n_jobs = 30


# orbit altitude:
Rsat = 500 # km
wavelengths_micron = [0.452, 1.6]  #0.672 , 1.6
sun_azimuth = 45
sun_zenith = 150
#azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
#zenith: Solar beam zenith angle in range (90,180]  

SATS_NUMBER_SETUP = 10 # satellites number to build the setup, for the inverse, we can use less satellites.
SATS_NUMBER_INVERSE = SATS_NUMBER_SETUP#10 # how much satelliets will be used for the inverse.

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""
MieTablesPath = os.path.abspath("./mie_tables")
mie_base_path = CALC_MIE_TABLES(MieTablesPath,wavelengths_micron,mie_options)
#mie_base_path = mie_base_path[0]
# for example, mie_base_path = './mie_tables/polydisperse/Water_672nm.scat'


if(isinstance(wavelengths_micron, list)):
    wavelengths_micron.sort()# just for convenience, let's have it sorted.
    wavelengths_string = functools.reduce(operator.add,[str(int(1e3*j))+"_" for j in wavelengths_micron]).rstrip('_')
    # forward_dir, where to save evrerything that is related to forward model:
    forward_dir = './experiments/polychromatic_unity_flux_active_sats_{}_LES_cloud_field_rico_Water_{}nm'.format(SATS_NUMBER_SETUP,wavelengths_string)
else: # if wavelengths_micron is scalar
    forward_dir = './experiments/polychromatic_unity_flux_active_sats_{}_LES_cloud_field_rico_Water_{}nm'.format(SATS_NUMBER_SETUP,int(1e3*wavelengths_micron))

# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name_base = 'active_sats_{}_easiest_rico32x37x26'.format(SATS_NUMBER_SETUP)
# Write intermediate TensorBoardX results into log_name.
# The provided string is added as a comment to the specific run.

# --------------------------------------------------------
# Similation flags
DOFORWARD = True
DOINVERSE = True

# ---------------------------------------------------------------
# -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
# ---------------------------------------------------------------

CloudFieldFile = '../synthetic_cloud_fields/jpl_les/rico32x37x26.txt'
AirFieldFile = './ancillary_data/AFGL_summer_mid_lat.txt' # Path to csv file which contains temperature measurements
atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile,AirFieldFile,
                             MieTablesPath,wavelengths_micron)
droplets = atmosphere.get_scatterer('cloud')

# -----------------------------------------------
# ---------Point the cameras and ----------------
# ---------Calculate camera footprint at nadir --
# -----------------------------------------------
atmospheric_grid = atmosphere.grid
dx = atmospheric_grid.dx
dy = atmospheric_grid.dy

nz = atmospheric_grid.nz
nx = atmospheric_grid.nx
ny = atmospheric_grid.ny

Lx = atmospheric_grid.bounding_box.xmax - atmospheric_grid.bounding_box.xmin
Ly = atmospheric_grid.bounding_box.ymax - atmospheric_grid.bounding_box.ymin
Lz = atmospheric_grid.bounding_box.zmax - atmospheric_grid.bounding_box.zmin
L = max(Lx,Ly)

Lz_droplets = droplets.grid.bounding_box.zmax - droplets.grid.bounding_box.zmin
dz = Lz_droplets/nz

#USED FOV, RESOLUTION and SAT_LOOKATS:
PIXEL_FOOTPRINT = 0.01 # km
fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
cny = int(np.floor(L/PIXEL_FOOTPRINT))
cnx = int(np.floor(L/PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]

# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if(IFTUNE_CAM):
    L = 1.5*L
    fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
    cny = int(np.floor(L/PIXEL_FOOTPRINT))
    cnx = int(np.floor(L/PIXEL_FOOTPRINT))    

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
print("fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov,cnx,cny))

print(20*"-")
print(20*"-")
print(20*"-")

print("Medium summary")
print("nx = {}, ny = {},nz ={}".format(nx,ny,nz))
print("dx = {}, dy = {},dz ={}".format(dx,dy,dz))
print("Lx = {}, Ly = {},Lz ={}".format(Lx,Ly,Lz))
x_min = atmospheric_grid.bounding_box.xmin
x_max = atmospheric_grid.bounding_box.xmax

y_min = atmospheric_grid.bounding_box.ymin
y_max = atmospheric_grid.bounding_box.ymax

z_min = atmospheric_grid.bounding_box.zmin
z_max = atmospheric_grid.bounding_box.zmax 
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
    Currently, in pyshdom we can define few cameras with different resolution in one setup.
    What we also whant to do is to define different bands and resolution to different sateliites.
    For now all the cameras are identicale and have the same spectral channels.
    Each camera has camera_wavelengths_list.
    """  
    if(np.isscalar(wavelengths_micron)):
        wavelengths_micron = [wavelengths_micron]
    setup_wavelengths_list = SATS_NUMBER_SETUP*[wavelengths_micron]
    # Calculate irradiance of the spesific wavelength:
    # use plank function:
    temp = 5900 #K
    L_TOA = []
    Cosain = np.cos(np.deg2rad((180-sun_zenith)))
    for wavelengths_per_view in setup_wavelengths_list:
        # loop over the dimensios of the views
        L_TOA_per_view = []
        for wavelength in wavelengths_per_view:
            # loop over the wavelengths is a view
            L_TOA_per_view.append(Cosain*6.8e-5*1e-9*CloudCT_setup.plank(1e-6*wavelength,temp)) # units fo W/(m^2))
        
        L_TOA.append(L_TOA_per_view)
    
    solar_flux_scale = L_TOA # the forward simulation will run with unity flux,
    # this is a scale that we should consider to apply on the output images.
    # Rigth now, lets skip it.
    
    # create CloudCT setup:
    CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(\
        SATS_NUMBER = SATS_NUMBER_SETUP, ORBIT_ALTITUDE = Rsat, \
        CAM_FOV = fov, CAM_RES = (cnx,cny), SAT_LOOKATS = SAT_LOOKATS, \
        SATS_WAVELENGTHS_LIST = setup_wavelengths_list, SOLAR_FLUX_LIST = solar_flux_scale, VISSETUP = VISSETUP)
    """ 
    How to randomly choose N views from the total views:
    for i in range(10):
        NEW = CloudCT_VIEWS.Random_choose_N(3)
        NEW.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)    
        mlab.show()
        
    The update of the solar fluxes per wavelength can be done also by:
    CloudCT_VIEWS.update_solar_irradiances(solar_flux_scale)
    """ 
    
    # ----------------------------------------------------------
    # ---------numerical & scene Parameters---------------------
    # ---------for RTE solver and initializtion of the solver---
    # ----------------------------------------------------------
    solar_fluxes = np.full_like(wavelengths_micron, 1.0)# unity flux
    split_accuracies = np.full_like(wavelengths_micron, 0.1) 
    surface_albedos = np.full_like(wavelengths_micron, 0.05) # later we must update it since it depends on the wavelegth.
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
    rte_solvers = shdom.RteSolverArray()
    
    for wavelength,split_accuracy,solar_flux,surface_albedo in \
        zip(wavelengths_micron,split_accuracies,solar_fluxes,surface_albedos):
        
    
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
    
    rte_solvers.solve(maxiter=100)
        
    
    # -----------------------------------------------
    # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
    # -----------------------------------------------
    """
    Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
    The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
    """
    CloudCT_VIEWS.update_measurements(sensor=shdom.RadianceSensor(), projection = CloudCT_VIEWS, rte_solver = rte_solvers, n_jobs=n_jobs)
    # see the rendered images:
    SEE_IMAGES = False
    if(SEE_IMAGES):
        CloudCT_VIEWS.show_measurements(compare_for_test = False)
        # don't use the compare_for_test =  True, it is not mature enougth.
        
        
        #It is a good place to pick the radiance_threshold = Threshold for the radiance to create a cloud mask.
        #So here, we see the original images and below we will see the images after the radiance_threshold: 
        # --------------------------------------------------
        #  ----------------try radiance_threshold value:----
        # --------------------------------------------------
        #radiance_threshold = 0.04 # Threshold for the radiance to create a cloud mask.
        radiance_threshold = [0.023,0.02] # Threshold for the radiance to create a cloud mask.
        # Threshold is either a scalar or a list of length of measurements.
        CloudCT_VIEWS.show_measurements(radiance_threshold=radiance_threshold,compare_for_test = False)    
        
        plt.show()
        
    
    # -----------------------------------------------
    # ---------SAVE EVERYTHING FOR THIS SETUP -------
    # -----------------------------------------------
    
    medium = atmosphere
    shdom.save_forward_model(forward_dir, medium, rte_solvers, CloudCT_VIEWS.measurements)
        
    print('DONE forwared simulation')


# -----------------------------------------------
# ---------SOLVE INVERSE ------------------------
# -----------------------------------------------
"""
Solve inverse problem:
"""
if(DOINVERSE):
    
    # load the measurments to see the rendered images:
    medium, solver, measurements = shdom.load_forward_model(forward_dir)
    # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    USED_CAMERA = measurements.camera
    RENDERED_IMAGES = measurements.images    
    THIS_MULTI_VIEW_SETUP = USED_CAMERA.projection
    
    # ---------what to optimize----------------------------
    radiance_threshold = [0.023,0.02] # check these values befor inverse.
    SEE_SETUP = False
    SEE_IMAGES = True
    MICROPHYSICS = True
    # ------------------------------------------------

    # show the mutli view setup if you want.
    if(SEE_SETUP):
        THIS_MULTI_VIEW_SETUP.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)    
        mlab.show()    
        
    
    # ----------------------------------------------------------
    
    """
    Work with the optimization:
    """
    
    if not MICROPHYSICS:
    
        # -----------------------------------------------
        # ---------SOLVE for extinction only  -----------
        # -----------------------------------------------    
        """
        Estimate extinction with (what is known?):
        1. ground truth phase function (use_forward_phase, with mie_base_path)
        2. grid (use_forward_grid)
        3. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
        4. known albedo (use_forward_albedo)
        5. rayleigh scattering (add_rayleigh)
        
        """
        log_name = "extinction_only_"+log_name_base
        # -----------------------------------------------
        # ---------------- inverse parameters:-----------
        # -----------------------------------------------
        
        stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                
        # Initialization and Medium parameters:
        extinction = 0.01 # init extinction of the generator
        # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
        init = 'Homogeneous'
        # The mie_base_path is defined at the begining of this script.
        INIT_USE = ' --init '+ init
        add_rayleigh = True
        use_forward_mask = False#True
        use_forward_grid = True
        use_forward_albedo = True
        use_forward_phase = True
        if_save_gt_and_carver_masks = True
        if_save_final3d = True
        
        GT_USE = ''
        GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
        GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
        GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
        GT_USE = GT_USE + ' --use_forward_albedo' if use_forward_albedo else GT_USE
        GT_USE = GT_USE + ' --use_forward_phase' if use_forward_phase else GT_USE
        GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
        GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
        
        # (use_forward_mask, use_forward_grid, use_forward_albedo, use_forward_phase):
        # Use the ground-truth things. This is an inverse crime which is 
        # usefull for debugging/development.    
        
            
        # The log_name defined above
        # Write intermediate TensorBoardX results into log_name.
        # The provided string is added as a comment to the specific run.
        
        # note that currently, the invers_dir = forward_dir.
        # In forward_dir, the forward modeling parameters are be saved.
        # If invers_dir = forward_dir:
        # The invers_dir directory will be used to save the optimization results and progress.
        
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
        OTHER_PARAMS = ' --input_dir ' + forward_dir + \
            ' --extinction ' + str(extinction) +\
            ' --log ' + log_name +\
            ' --air_path ' + AirFieldFile +\
            ' --n_jobs '+ str(n_jobs)+\
            ' --loss_type '+ str(loss_type)+\
            ' --maxls '+ str(maxls)+\
            ' --maxiter '+ str(maxiter)+\
            ' --radiance_threshold '+ str(radiance_threshold)
        
        OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
        
        # We have: ground_truth, rte_solver, measurements.
        SCRIPTS_PATH = '../scripts'
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_extinction_lbfgs.py')+\
            OTHER_PARAMS + GT_USE + INIT_USE
        
        Optimaize1 = subprocess.call( cmd, shell=True)
    
    
    else: # if we here, we rec. MICROPHYSICS, but which?
        REC_only_lwc = False
        REC_only_reff = False
        REC_only_veff = False
        REC_all = True
        if(REC_only_lwc):
            
            # -----------------------------------------------
            # ---------SOLVE for lwc only  ------------------
            # -----------------------------------------------    
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and variance, Hence,
               the albedo should be known.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "lwc_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            lwc = 0.01 # init lwc of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = False#True
            use_forward_grid = True
            use_forward_reff = True
            use_forward_veff = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_reff' if use_forward_reff else GT_USE
            GT_USE = GT_USE + ' --use_forward_veff' if use_forward_veff else GT_USE                
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
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --lwc ' + str(lwc) +\
                ' --log ' + log_name +\
                ' --air_path ' + AirFieldFile +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)            
        
        
        if(REC_only_reff):
            
            # -----------------------------------------------
            # ---------SOLVE for reff only  ------------------
            # -----------------------------------------------    
            """
            Estimate reff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective variance and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "reff_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            reff = 15 # init reff of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = False#True
            use_forward_grid = True
            use_forward_lwc = True
            use_forward_veff = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_lwc' if use_forward_lwc else GT_USE
            GT_USE = GT_USE + ' --use_forward_veff' if use_forward_veff else GT_USE                
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
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --reff ' + str(reff) +\
                ' --log ' + log_name +\
                ' --air_path ' + AirFieldFile +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)  
            
        if(REC_only_veff):
            
            # -----------------------------------------------
            # ---------SOLVE for veff only  ------------------
            # -----------------------------------------------    
            """
            Estimate veff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "veff_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            veff = 0.19 # init veff of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = True#True
            use_forward_grid = True
            use_forward_reff = True
            use_forward_lwc = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_lwc' if use_forward_lwc else GT_USE
            GT_USE = GT_USE + ' --use_forward_reff' if use_forward_reff else GT_USE                
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
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --veff ' + str(veff) +\
                ' --log ' + log_name +\
                ' --air_path ' + AirFieldFile +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)  
            
              
        if(REC_all):
            
            # -----------------------------------------------
            # ---------SOLVE for lwc only  ------------------
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
            add_rayleigh = True
            use_forward_mask = True#True
            use_forward_grid = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --use_forward_veff --lwc_scaling 15 --reff_scaling 0.01'# -----------------                       
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE               
            GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
            GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
            
            # optimization:
            globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
            maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            maxls = 100 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
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
                ' --air_path ' + AirFieldFile +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)
                #' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)            
              
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
