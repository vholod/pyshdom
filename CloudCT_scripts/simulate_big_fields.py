
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
from shdom.CloudCT_Utils import *

# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 


VISSETUP = False

# -----------------------------------------------------------------
# mia table parameters:
start_reff = 0.01
end_reff = 25.0
start_veff = 0.01
end_veff = 0.7
radius_cutoff = 65.0
mie_options = {
    'start_reff': start_reff,# Starting effective radius [Micron]
    'end_reff': end_reff,
    'num_reff': 100,
    'start_veff': start_veff,
    'end_veff': end_veff,
    'num_veff': 100,
    'radius_cutoff': radius_cutoff # The cutoff radius for the pdf averaging [Micron]
}


# orbit altitude:
Rsat = 500 # km
GSD = 0.5*0.02 # in km, it is the ground spatial resolution.
wavelengths_micron = [0.672]  #0.672 , 1.6
sun_azimuth = 45
sun_zenith = 155
maxiter = 150
n_jobs = 60

#azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
#zenith: Solar beam zenith angle in range (90,180]  

SATS_NUMBER_SETUP = 10 # satellites number to build the setup, for the inverse, we can use less satellites.

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""
MieTablesPath = os.path.abspath("./mie_tables")
mie_base_path = CALC_MIE_TABLES(MieTablesPath,wavelengths_micron,mie_options)
#mie_base_path = mie_base_path[0]
# for example, mie_base_path = './mie_tables/polydisperse/Water_672nm.scat'



# ---------------------------------------------------------------
# -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
# ---------------------------------------------------------------
#CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_256x256x100_5000CCN_50m_micro_256_0000007620.txt" # works but slow.
#CloudFieldFile = "/wdata/vadim/CloudGeometry/clouds from Udi/16samples/BOMEX_512x512x200_20m_20m_1s_512_0000007360_2_3_old_pyshdom.txt"
CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/BOMEX_21600_515x515x71_6520_old_pyshdom.txt"
#CloudFieldFile = "/wdata/clouds_from_weizmann/BOMEX_512X512X170_500CCN_20m/cropped_for_new_pyshdom/danny_small_cloud_old_pyshdom.txt"



AirFieldFile = '../ancillary_data/AFGL_summer_mid_lat.txt' # Path to csv file which contains temperature measurements
    

atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile=CloudFieldFile, AirFieldFile = AirFieldFile, \
               air_num_points = 20, air_max_alt = 20.0, MieTablesPath=MieTablesPath, 
               wavelengths_micron=wavelengths_micron,wavelength_averaging=False)


fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0),
                   size=(400, 400))

atmosphere.show_scatterer(name='cloud')


values = np.array(1*[0.0])
atmosphere.pad_scatterer(name = 'cloud', axis=0, right = True, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=0, right = False, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=1, right = True, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=1, right = False, values=values) 

#atmosphere.pad_scatterer(name = 'cloud', axis=2, right = True, values=values) 
#atmosphere.pad_scatterer(name = 'cloud', axis=2, right = False, values=values)

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
PIXEL_FOOTPRINT = GSD # km
fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
cny = int(np.floor(L/PIXEL_FOOTPRINT))
cnx = int(np.floor(L/PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]

# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if(IFTUNE_CAM):
    L = 1.2*L
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

 
# ---------------------------------------------------------------
# ---------------CREATE THE SETUP----------------------------
# ---------------------------------------------------------------
VISSETUP  = True
sat_positions, near_nadir_view_index = CloudCT_setup.StringOfPearls(SATS_NUMBER = SATS_NUMBER_SETUP,\
                                                                    orbit_altitude = Rsat)

names = ["sat"+str(i+1) for i in range(len(sat_positions))]                 

# we intentialy, work with projections lists.
up_list = np.array(len(sat_positions)*[0,1,0]).reshape(-1,3) # default up vector per camera.
projection = shdom.MultiViewProjection()
for pos,lookat,up,name in zip(sat_positions,\
                          SAT_LOOKATS,up_list,names):
    
    x,y,z = pos
    loop_projection = shdom.PerspectiveProjection(fov, cnx, cny, x, y, z)

    loop_projection.look_at_transform(lookat, up)
                    
    projection.add_projection(loop_projection,name)
    """
    Each projection has:
    ['x', 'y', 'z', 'mu', 'phi'] per pixel and npix, names, resolution.
    """

    
# visualization params:
scale = 500
axisWidth = 0.02
axisLenght = 500    
FullCone = True
if(VISSETUP):
    for view,view_name in zip(projection._projection_list,names):
        view.show_camera(scale,axisWidth,axisLenght,FullCone) 
        t = view.position
        mlab.text3d(t[0]+0.2*scale, t[1], t[2]+0.2*scale, view_name, color=(1,0,0),scale=0.02*scale)
                
    figh = mlab.gcf()
    mlab.orientation_axes(figure=figh)
    mlab.show()



    
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

# ----------------------------------------------------------
# ---------numerical & scene Parameters---------------------
# ---------for RTE solver and initializtion of the solver---
# ----------------------------------------------------------
solar_fluxes = np.full_like(wavelengths_micron, 1.0)# unity flux
split_accuracies = np.full_like(wavelengths_micron, 0.1) 
surface_albedos = np.full_like(wavelengths_micron, 0.05) # later we must update it since it depends on the wavelegth.
# split_accuracy of 0.1 gives nice results, For the rico cloud i didn't see that decreasing the split accuracy improves the rec.
# so fo the rico loud Let's use split_accuracy = 0.1.
num_mu = 8
num_phi = 16
# note: whae num_mu, num_phi are 16,32, the retrievals look better than with 8,16.
adapt_grid_factor = 5
solution_accuracy = 0.0001
max_total_mb = 1000000.0
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
rte_solution_file = CloudFieldFile.split('/')[-1].split('.')[0]+'_solution.pkl'
rte_solvers.solve(maxiter=maxiter)
#if os.path.exists(rte_solution_file):
    ## load the solution:
    #file = open(rte_solution_file, 'rb')
    #data = file.read()
    #file.close()
    #rte_solvers = pickle.loads(data)    
#else:
    ##solve it:
    #rte_solvers.solve(maxiter=maxiter)
    ## save the solution:
    #file = open(rte_solution_file, 'wb')
    #file.write(pickle.dumps(rte_solvers, -1))
    #file.close()    
    

        

# -----------------------------------------------
# ---------RENDER IMAGES FOR CLOUDCT SETUP ------
# -----------------------------------------------
"""
Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
"""

from collections import OrderedDict

sensor = shdom.RadianceSensor() # Currently, this method sopports only sensor of type shdom.RadianceSensor().
camera = shdom.Camera(sensor, projection)
# render all view per wavelength:
Radiances = camera.render(rte_solvers, n_jobs=n_jobs)   
IMAGES_DICT = OrderedDict()
# see the rendered images:
SEE_IMAGES = True
if(SEE_IMAGES):
    
    ncols = 5 # colums, in subplots
    nrows_ncols = [int(np.ceil(projection.num_projections/ncols)), int(ncols)]
    
    images_array = np.array(Radiances)
    MAXI = images_array.max()
        
    fig = plt.figure(figsize=(20, 20))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )  
    
    # show all
    for projection_index, (ax, name) in enumerate(zip(grid, names)):
        image = Radiances[projection_index].copy()
        IMAGES_DICT[name] = image
        ax.set_axis_off()
        im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
        ax.set_title("{}".format(name))
    
    grid.cbar_axes[0].colorbar(im) 
    
    
file_name = CloudFieldFile.split('/')[-1].split('.')[0]+'.mat'
sio.savemat(file_name, IMAGES_DICT)
plt.show()    


print('DONE forwared simulation')