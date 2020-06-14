import os
import sys
# import mayavi.mlab as mlab
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
from shdom.CloudCT_Utils import *

# importing functools for reduce() 
import functools
# importing operator for operator functions 
import operator
import yaml

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# Load run parameters
with open("run_params.yaml", 'r') as f:
    run_params = yaml.full_load(f)

mie_options = run_params['mie_options']  # mie params
viz_options = run_params['viz_options']  # visualization params
n_jobs = run_params['n_jobs']

Rsat = run_params['Rsat']
GSD = run_params['GSD']
wavelengths_micron = run_params['wavelengths_micron']
sun_azimuth = run_params['sun_azimuth']
sun_zenith = run_params['sun_zenith']

SATS_NUMBER_SETUP = run_params['SATS_NUMBER_SETUP']
SATS_NUMBER_INVERSE = run_params['SATS_NUMBER_INVERSE']

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""
MieTablesPath = os.path.abspath("./mie_tables")
mie_base_path = CALC_MIE_TABLES(MieTablesPath, wavelengths_micron, mie_options)
# mie_base_path = mie_base_path[0]
# for example, mie_base_path = './mie_tables/polydisperse/Water_672nm.scat'

middle_dir_name = f'unity_flux_active_sats_{SATS_NUMBER_SETUP}_GSD_{int(1e3 * GSD)}m_LES_cloud_field_rico_Water'
if isinstance(wavelengths_micron, list):

    wavelengths_micron.sort()  # just for convenience, let's have it sorted.
    wavelengths_string = functools.reduce(operator.add, [str(int(1e3 * j)) + "_" for j in wavelengths_micron]).rstrip('_')

    # forward_dir, where to save everything that is related to forward model:
    forward_dir = f'./experiments/polychromatic_{middle_dir_name}_{wavelengths_string}nm'

else:  # if wavelengths_micron is scalar
    forward_dir = f'./experiments/monochromatic_{middle_dir_name}_{int(1e3 * wavelengths_micron)}nm'

# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name_base = f'active_sats_{SATS_NUMBER_SETUP}_easiest_rico32x37x26'
# Write intermediate TensorBoardX results into log_name.
# The provided string is added as a comment to the specific run.


# -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
# Path to csv file which contains temperature measurements or None if the atmosphere will not consider any air.
AirFieldFile = run_params['AirFieldFile'] if not viz_options['CENCEL_AIR'] else None

atmosphere = CloudCT_setup.Prepare_Medium(run_params['CloudFieldFile'], AirFieldFile,
                                          MieTablesPath, wavelengths_micron)
droplets = atmosphere.get_scatterer('cloud')

# --------- Point the cameras and Calculate camera footprint at nadir ---------
atmospheric_grid = atmosphere.grid
dx = atmospheric_grid.dx
dy = atmospheric_grid.dy

nz = atmospheric_grid.nz
nx = atmospheric_grid.nx
ny = atmospheric_grid.ny

Lx = atmospheric_grid.bounding_box.xmax - atmospheric_grid.bounding_box.xmin
Ly = atmospheric_grid.bounding_box.ymax - atmospheric_grid.bounding_box.ymin
Lz = atmospheric_grid.bounding_box.zmax - atmospheric_grid.bounding_box.zmin
L = max(Lx, Ly)

Lz_droplets = droplets.grid.bounding_box.zmax - droplets.grid.bounding_box.zmin
dz = Lz_droplets / nz

# USED FOV, RESOLUTION and SAT_LOOKATS:
PIXEL_FOOTPRINT = GSD  # km
fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
cny = int(np.floor(L / PIXEL_FOOTPRINT))
cnx = int(np.floor(L / PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

# Sometimes it is more convenient to use wide fov to see the whole cloud from all the view points.
# so the FOV is also tuned:
# -- TUNE FOV, CNY,CNX:
if run_params['IFTUNE_CAM']:
    L = 1.5 * L
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
    cny = int(np.floor(L / PIXEL_FOOTPRINT))
    cnx = int(np.floor(L / PIXEL_FOOTPRINT))

# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variable LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
if run_params['IFTUNE_CAM']:
    LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1, 3)  # currently, all satellites lookat the same point.

print(20 * "-")
print(20 * "-")
print(20 * "-")

print("CAMERA intrinsics summary")
print(f"fov = {fov}[deg], cnx = {cnx}[pixels],cny ={cny}[pixels]")

print(20 * "-")
print(20 * "-")
print(20 * "-")

print("Medium summary")
print(f"nx = {nx}, ny = {ny},nz ={nz}")
print(f"dx = {dx}, dy = {dy},dz ={dz}")
print(f"Lx = {Lx}, Ly = {Ly},Lz ={Lz}")

x_min = atmospheric_grid.bounding_box.xmin
x_max = atmospheric_grid.bounding_box.xmax

y_min = atmospheric_grid.bounding_box.ymin
y_max = atmospheric_grid.bounding_box.ymax

z_min = atmospheric_grid.bounding_box.zmin
z_max = atmospheric_grid.bounding_box.zmax

print(f"xmin = {x_min}, ymin = {y_min},zmin ={z_min}")
print(f"xmax = {x_max}, ymax = {y_max},zmax ={z_max}")

print(20 * "-")
print(20 * "-")
print(20 * "-")

if run_params['DOFORWARD']:
    forward_options = run_params['forward_options']

    # ---------------------------------------------------------------
    # ---------------CREATE THE SETUP----------------------------
    # ---------------------------------------------------------------
    """
    Currently, in pyshdom we can define few cameras with different resolution in one setup.
    What we also whant to do is to define different bands and resolution to different sateliites.
    For now all the cameras are identicale and have the same spectral channels.
    Each camera has camera_wavelengths_list.
    """
    if np.isscalar(wavelengths_micron):
        wavelengths_micron = [wavelengths_micron]

    setup_wavelengths_list = SATS_NUMBER_SETUP * [wavelengths_micron]
    # Calculate irradiance of the specific wavelength:
    # use plank function:
    L_TOA = []
    Cosain = np.cos(np.deg2rad((180 - sun_zenith)))
    for wavelengths_per_view in setup_wavelengths_list:
        # loop over the dimensios of the views
        L_TOA_per_view = []
        for wavelength in wavelengths_per_view:
            # loop over the wavelengths is a view
            L_TOA_per_view.append(Cosain * 6.8e-5 * 1e-9 * CloudCT_setup.plank(1e-6 * wavelength, forward_options['temp']))  # units fo W/(m^2))

        L_TOA.append(L_TOA_per_view)

    solar_flux_scale = L_TOA  # the forward simulation will run with unity flux,
    # this is a scale that we should consider to apply on the output images.
    # Right now, lets skip it.

    # create CloudCT setup:
    CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(
        SATS_NUMBER=SATS_NUMBER_SETUP, ORBIT_ALTITUDE=Rsat,
        CAM_FOV=fov, CAM_RES=(cnx, cny), SAT_LOOKATS=SAT_LOOKATS,
        SATS_WAVELENGTHS_LIST=setup_wavelengths_list, SOLAR_FLUX_LIST=solar_flux_scale,
        VISSETUP=viz_options['VISSETUP'])
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

    # ---------numerical & scene Parameters---------------------
    # ---------for RTE solver and initializtion of the solver---
    solar_fluxes = np.full_like(wavelengths_micron, run_params['solar_fluxes_val'])  # unity flux
    split_accuracies = np.full_like(wavelengths_micron, run_params['split_accuracies_val'])
    surface_albedos = np.full_like(wavelengths_micron, run_params['surface_albedos_val'])

    adapt_grid_factor = 5  # TODO not in use
    solution_accuracy = 0.0001  # TODO not in use

    # Generate a solver array for a multispectral solution.
    # it is great that we can use the parallel solution of all solvers.
    rte_solvers = shdom.RteSolverArray()

    for wavelength, split_accuracy, solar_flux, surface_albedo in \
            zip(wavelengths_micron, split_accuracies, solar_fluxes, surface_albedos):

        numerical_params = shdom.NumericalParameters(num_mu_bins=forward_options['num_mu'],
                                                     num_phi_bins=forward_options['num_phi'],
                                                     split_accuracy=split_accuracy,
                                                     max_total_mb=forward_options['max_total_mb'])

        scene_params = shdom.SceneParameters(
            wavelength=wavelength,
            surface=shdom.LambertianSurface(albedo=surface_albedo),
            source=shdom.SolarSource(azimuth=sun_azimuth, zenith=sun_zenith, flux=solar_flux))

        # ---------initilize an RteSolver object---------    
        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)
        rte_solvers.add_solver(rte_solver)

    # ---------RTE SOLVE ----------------------------
    rte_solvers.solve(maxiter=forward_options['rte_solver_max_iter'])

    # -----------------------------------------------
    # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
    # -----------------------------------------------
    """
    Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
    The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
    """
    CloudCT_VIEWS.update_measurements(sensor=shdom.RadianceSensor(), projection=CloudCT_VIEWS, rte_solver=rte_solvers,
                                      n_jobs=n_jobs)
    # see the rendered images:
    SEE_IMAGES = False
    if SEE_IMAGES:
        CloudCT_VIEWS.show_measurements(compare_for_test=False)
        # don't use the compare_for_test =  True, it is not mature enough.
        # It is a good place to pick the radiance_threshold = Threshold for the radiance to create a cloud mask.
        # So here, we see the original images and below we will see the images after the radiance_threshold:
        # --------------------------------------------------
        #  ----------------try radiance_threshold value:----
        # --------------------------------------------------
        CloudCT_VIEWS.show_measurements(radiance_threshold=run_params['radiance_threshold'], compare_for_test=False)

        plt.show()

    # ---------SAVE EVERYTHING FOR THIS SETUP -------
    medium = atmosphere
    shdom.save_forward_model(forward_dir, medium, rte_solvers, CloudCT_VIEWS.measurements)

    print('DONE forward simulation')

# ---------SOLVE INVERSE ------------------------
if run_params['DOINVERSE']:
    inverse_options = run_params['inverse_options']

    # load the measurments to see the rendered images:
    medium, solver, measurements = shdom.load_forward_model(forward_dir)
    # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    USED_CAMERA = measurements.camera
    RENDERED_IMAGES = measurements.images
    THIS_MULTI_VIEW_SETUP = USED_CAMERA.projection

    SEE_IMAGES = True  # TODO not in use

    # show the mutli view setup if you want.
    if inverse_options['SEE_SETUP']:
        THIS_MULTI_VIEW_SETUP.show_setup(scale=viz_options['scale'], axisWidth=viz_options['axisWidth'],
                                         axisLenght=viz_options['axisLenght'], FullCone=True)
        # figh = mlab.gcf()
        # mlab.orientation_axes(figure=figh)
        # mlab.show()

    run_type = inverse_options['recover_type'] if inverse_options['MICROPHYSICS'] else 'extinction'

    log_name = run_type + "_only_" + log_name_base

    INIT_USE = ' --init ' + inverse_options['init']

    GT_USE = ''
    GT_USE = GT_USE + ' --add_rayleigh' if inverse_options['add_rayleigh'] and not viz_options['CENCEL_AIR'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_mask' if inverse_options['use_forward_mask'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_grid' if inverse_options['use_forward_grid'] else GT_USE
    GT_USE = GT_USE + ' --save_gt_and_carver_masks' if inverse_options['if_save_gt_and_carver_masks'] else GT_USE
    GT_USE = GT_USE + ' --save_final3d' if inverse_options['if_save_final3d'] else GT_USE

    # The mie_base_path is defined at the beginning of this script.
    # (use_forward_mask, use_forward_grid, use_forward_albedo, use_forward_phase):
    # Use the ground-truth things. This is an inverse crime which is
    # useful for debugging/development.

    # The log_name defined above
    # Write intermediate TensorBoardX results into log_name.
    # The provided string is added as a comment to the specific run.

    # note that currently, the invers_dir = forward_dir.
    # In forward_dir, the forward modeling parameters are be saved.
    # If inverse_dir = forward_dir:
    # The inverse_dir directory will be used to save the optimization results and progress.

    OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                   ' --log ' + log_name + \
                   ' --n_jobs ' + str(n_jobs) + \
                   ' --loss_type ' + inverse_options['loss_type'] + \
                   ' --maxls ' + str(inverse_options['maxls']) + \
                   ' --maxiter ' + str(inverse_options['maxiter'])

    OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if inverse_options['globalopt'] else OTHER_PARAMS

    OTHER_PARAMS = OTHER_PARAMS + ' --air_path ' + AirFieldFile if not viz_options['CENCEL_AIR'] else OTHER_PARAMS

    OTHER_PARAMS = OTHER_PARAMS + ' --radiance_threshold ' + " ".join(map(str, run_params['radiance_threshold'])) if run_type != 'reff_and_lwc' else OTHER_PARAMS

    if inverse_options['MICROPHYSICS']:
        # -----------------------------------------------
        # ---------SOLVE for lwc only  ------------------
        # -----------------------------------------------
        if run_type == 'lwc':
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and variance, Hence,
               the albedo should be known.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
    
            """

            GT_USE += ' --use_forward_reff'
            GT_USE += ' --use_forward_veff'
            OTHER_PARAMS += ' --lwc ' + str(inverse_options['lwc'])

        # -----------------------------------------------
        # ---------SOLVE for reff only  ------------------
        # -----------------------------------------------
        elif run_type == 'reff':
            """
            Estimate reff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective variance and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
            """

            GT_USE += ' --use_forward_lwc'
            GT_USE += ' --use_forward_veff'
            OTHER_PARAMS += ' --reff ' + str(inverse_options['reff'])

        # -----------------------------------------------
        # ---------SOLVE for veff only  ------------------
        # -----------------------------------------------
        elif run_type == 'veff':
            """
            Estimate veff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
    
            """

            GT_USE += ' --use_forward_lwc'
            GT_USE += ' --use_forward_reff'
            OTHER_PARAMS += ' --veff ' + str(inverse_options['veff'])


        # -----------------------------------------------
        # ---------SOLVE for lwc and reff  ------------------
        # -----------------------------------------------
        elif run_type == 'reff_and_lwc':
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. rayleigh scattering (add_rayleigh)
            4. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
    
            """

            GT_USE += ' --use_forward_veff'
            GT_USE += ' --lwc_scaling ' + str(inverse_options['lwc_scaling_val'])
            GT_USE += ' --reff_scaling ' + str(inverse_options['reff_scaling_val'])
            OTHER_PARAMS += ' --reff ' + str(inverse_options['reff'])
            OTHER_PARAMS += ' --lwc ' + str(inverse_options['lwc'])
    # -----------------------------------------------
    # ---------SOLVE for extinction only  -----------
    # -----------------------------------------------
    else:
        """
        Estimate extinction with (what is known?):
        1. ground truth phase function (use_forward_phase, with mie_base_path)
        2. grid (use_forward_grid)
        3. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
        4. known albedo (use_forward_albedo)
        5. rayleigh scattering (add_rayleigh)

        """

        GT_USE += ' --use_forward_albedo'
        GT_USE += ' --use_forward_phase'
        OTHER_PARAMS += ' --extinction ' + str(inverse_options['extinction'])

    optimizer_path = inverse_options['microphysics_optimizer'] if inverse_options['MICROPHYSICS'] else inverse_options['extinction_optimizer']

    # We have: ground_truth, rte_solver, measurements.

    cmd = 'python ' + \
          os.path.join(inverse_options['scripts_path'], optimizer_path) + \
          OTHER_PARAMS + \
          GT_USE + \
          INIT_USE

    Optimize1 = subprocess.call(cmd, shell=True)

    # Time to show the results in 3D visualization:
    if inverse_options['VIS_RESULTS3D']:
        """
        The forward_dir id a folder that containes:
        medium, solver, measurements.
        They loaded before. To see the final state, the medium is not 
        enough, the medium_estimator is needed.
        load the measurments to see the rendered images:
        """

        # what state to load? I prefere the last one!
        logs_dir = os.path.join(forward_dir, 'logs')
        logs_prefix = os.path.join(logs_dir, log_name)
        logs_files = glob.glob(logs_prefix + '-*')

        times = [i.split(f'{int(1e3 * wavelength_micron)}-')[-1] for i in logs_files]
        # sort the times to find the last one.
        timestamp = [time.mktime(time.strptime(i, "%d-%b-%Y-%H:%M:%S")) for i in times]
        # time.mktime(t) This is the inverse function of localtime()
        timestamp.sort()
        timestamp = [time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime(i)) for i in timestamp]
        # now, the timestamp are sorted, and I want the last stamp to visualize.
        connector = f'{int(1e3 * wavelength_micron)}-'
        log2load = logs_prefix + '-' + timestamp[-1]
        # print here the Final results files:
        Final_results_3Dfiles = glob.glob(log2load + '/FINAL_3D_*.mat')
        print(f"{len(Final_results_3Dfiles)} files with the results in 3D were created:")
        for _file in Final_results_3Dfiles:
            print(_file)

        # ---------------------------------------------------------

        # Don't want to use it now, state2load = os.path.join(log2load,'final_state.ckpt')

        print(f"use tensorboard --logdir {log2load} --bind_all")

    print("done")
