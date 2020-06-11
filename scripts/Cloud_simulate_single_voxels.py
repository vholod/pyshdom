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
import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from CloudCT_Utils import *
from render_radiance_toa import *

# importing functools for reduce()
import functools
# importing operator for operator functions
import operator

# -----------------------------------------------------------------
# -----------------------------------------------------------------

# visualization params:
VISSETUP = False
scale = 500
axisWidth = 0.02
axisLenght = 5000

# sat number:
# SATS_NUMBER = 10
n_jobs = 30

# orbit altitude:
Rsat = 500  # km
wavelengths_micron = [0.672, 1.6]  # 0.672 , 1.6
sun_azimuth = 45
sun_zenith = 150
# azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
# zenith: Solar beam zenith angle in range (90,180]

SATS_NUMBER_SETUP = 10  # satellites number to build the setup, for the inverse, we can use less satellites.
SATS_NUMBER_INVERSE = SATS_NUMBER_SETUP  # 10 # how much satellites will be used for the inverse.

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""

if (isinstance(wavelengths_micron, list)):
    wavelengths_micron.sort()  # just for convenience, let's have it sorted.
    wavelengths_string = functools.reduce(operator.add, [str(int(1e3 * j)) + "_" for j in wavelengths_micron]).rstrip(
        '_')
    # forward_dir, where to save evrerything that is related to forward model:
    forward_dir = './experiments/polychromatic_single_voxel_{}nm'.format(
        SATS_NUMBER_SETUP, wavelengths_string)
else:  # if wavelengths_micron is scalar
    forward_dir = './experiments/monochromatic_single_voxel_{}nm'.format(
        SATS_NUMBER_SETUP, int(1e3 * wavelengths_micron))

# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name_base = 'active_sats_{}_single'.format(SATS_NUMBER_SETUP)
# Write intermediate TensorBoardX results into log_name.
# The provided string is added as a comment to the specific run.


# Microphysical grid definition
reff_range = np.linspace(6.0, 18.5, 3)
lwc_range = np.linspace(1e-2, 0.85, 3)
# ---------------------------------------------------------------
# -------------create single voxel cloud--------------------------------
# ---------------------------------------------------------------
# Run and save the results
stokes = generate_measurements(wavelengths_micron, lwc_range, reff_range,
                               view_zenith, view_azimuth, solar_zenith, solar_azimuth)



CloudFieldFile = '../synthetic_cloud_fields/jpl_les/rico32x37x26.txt'


AirFieldFile = './ancillary_data/AFGL_summer_mid_lat.txt'  # Path to csv file which contains temperature measurements
atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile, AirFieldFile,
                                          MieTablesPath, wavelengths_micron)
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
L = max(Lx, Ly)

Lz_droplets = droplets.grid.bounding_box.zmax - droplets.grid.bounding_box.zmin
dz = Lz_droplets / nz

# USED FOV, RESOLUTION and SAT_LOOKATS:
PIXEL_FOOTPRINT = 0.01  # km
fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
cny = int(np.floor(L / PIXEL_FOOTPRINT))
cnx = int(np.floor(L / PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if (IFTUNE_CAM):
    L = 1.5 * L
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
    cny = int(np.floor(L / PIXEL_FOOTPRINT))
    cnx = int(np.floor(L / PIXEL_FOOTPRINT))

# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variavle LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
if (IFTUNE_CAM):
    LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1, 3)  # currently, all satellites lookat the same point.

print(20 * "-")
print(20 * "-")
print(20 * "-")

print("CAMERA intrinsics summary")
print("fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov, cnx, cny))

print(20 * "-")
print(20 * "-")
print(20 * "-")

print("Medium summary")
print("nx = {}, ny = {},nz ={}".format(nx, ny, nz))
print("dx = {}, dy = {},dz ={}".format(dx, dy, dz))
print("Lx = {}, Ly = {},Lz ={}".format(Lx, Ly, Lz))
x_min = atmospheric_grid.bounding_box.xmin
x_max = atmospheric_grid.bounding_box.xmax

y_min = atmospheric_grid.bounding_box.ymin
y_max = atmospheric_grid.bounding_box.ymax

z_min = atmospheric_grid.bounding_box.zmin
z_max = atmospheric_grid.bounding_box.zmax
print("xmin = {}, ymin = {},zmin ={}".format(x_min, y_min, z_min))
print("xmax = {}, ymax = {},zmax ={}".format(x_max, y_max, z_max))

print(20 * "-")
print(20 * "-")
print(20 * "-")

if (DOFORWARD):
    # ---------------------------------------------------------------
    # ---------------CREATE THE SETUP----------------------------
    # ---------------------------------------------------------------
    """
    Currently, in pyshdom we can define few cameras with different resolution in one setup.
    What we also whant to do is to define different bands and resolution to different sateliites.
    For now all the cameras are identicale and have the same spectral channels.
    Each camera has camera_wavelengths_list.
    """
    if (np.isscalar(wavelengths_micron)):
        wavelengths_micron = [wavelengths_micron]
    setup_wavelengths_list = SATS_NUMBER_SETUP * [wavelengths_micron]
    # Calculate irradiance of the spesific wavelength:
    # use plank function:
    temp = 5900  # K
    L_TOA = []
    Cosain = np.cos(np.deg2rad((180 - sun_zenith)))
    for wavelengths_per_view in setup_wavelengths_list:
        # loop over the dimensios of the views
        L_TOA_per_view = []
        for wavelength in wavelengths_per_view:
            # loop over the wavelengths is a view
            L_TOA_per_view.append(
                Cosain * 6.8e-5 * 1e-9 * CloudCT_setup.plank(1e-6 * wavelength, temp))  # units fo W/(m^2))

        L_TOA.append(L_TOA_per_view)

    solar_flux_scale = L_TOA  # the forward simulation will run with unity flux,
    # this is a scale that we should consider to apply on the output images.
    # Rigth now, lets skip it.

    # create CloudCT setup:
    CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create( \
        SATS_NUMBER=SATS_NUMBER_SETUP, ORBIT_ALTITUDE=Rsat, \
        CAM_FOV=fov, CAM_RES=(cnx, cny), SAT_LOOKATS=SAT_LOOKATS, \
        SATS_WAVELENGTHS_LIST=setup_wavelengths_list, SOLAR_FLUX_LIST=solar_flux_scale, VISSETUP=VISSETUP)
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
    solar_fluxes = np.full_like(wavelengths_micron, 1.0)  # unity flux
    split_accuracies = np.full_like(wavelengths_micron, 0.1)
    surface_albedos = np.full_like(wavelengths_micron,
                                   0.05)  # later we must update it since it depends on the wavelegth.
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

    for wavelength, split_accuracy, solar_flux, surface_albedo in \
            zip(wavelengths_micron, split_accuracies, solar_fluxes, surface_albedos):
        numerical_params = shdom.NumericalParameters(num_mu_bins=num_mu, num_phi_bins=num_phi,
                                                     split_accuracy=split_accuracy, max_total_mb=max_total_mb)

        scene_params = shdom.SceneParameters(
            wavelength=wavelength,
            surface=shdom.LambertianSurface(albedo=surface_albedo),
            source=shdom.SolarSource(azimuth=sun_azimuth, zenith=sun_zenith, flux=solar_flux)
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
    CloudCT_VIEWS.update_measurements(sensor=shdom.RadianceSensor(), projection=CloudCT_VIEWS, rte_solver=rte_solvers,
                                      n_jobs=n_jobs)
    # see the rendered images:
    SEE_IMAGES = False
    if (SEE_IMAGES):
        CloudCT_VIEWS.show_measurements(compare_for_test=False)
        # don't use the compare_for_test =  True, it is not mature enougth.

        # It is a good place to pick the radiance_threshold = Threshold for the radiance to create a cloud mask.
        # So here, we see the original images and below we will see the images after the radiance_threshold:
        # --------------------------------------------------
        #  ----------------try radiance_threshold value:----
        # --------------------------------------------------
        # radiance_threshold = 0.04 # Threshold for the radiance to create a cloud mask.
        radiance_threshold = [0.023, 0.02]  # Threshold for the radiance to create a cloud mask.
        # Threshold is either a scalar or a list of length of measurements.
        CloudCT_VIEWS.show_measurements(radiance_threshold=radiance_threshold, compare_for_test=False)

        plt.show()

    # -----------------------------------------------
    # ---------SAVE EVERYTHING FOR THIS SETUP -------
    # -----------------------------------------------

    medium = atmosphere
    shdom.save_forward_model(forward_dir, medium, rte_solvers, CloudCT_VIEWS.measurements)

    print('DONE forwared simulation')