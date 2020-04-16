
import os 
#import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom


mie = shdom.MiePolydisperse()
mie.read_table(file_path='mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
#droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
#droplets.load_from_csv('../synthetic_cloud_fields/small_cloud_les/view55_small.txt', veff=0.1)
#droplets.load_from_csv('../synthetic_cloud_fields/small_cloud_les/cut_from_dannys_clouds_S1.txt', veff=0.1)
droplets.load_from_csv('../synthetic_cloud_fields/shdom/BOMEX_128x128x100_2000CCN_50m_micro_128_0000007200.txt', veff=0.1)


droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('./ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, 4, 40))
#air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
rayleigh = shdom.Rayleigh(wavelength=0.672)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()


"""
Generate an Medium with two type of scatterers and initilize an RteSolver object. 
This will initialize all of shdom internal structures and grids.

SceneParameters() contains parameters such as surface albedo and solar radiance
NumericalParameters() contains parameters such as number of angular bins and split cell ratio.
All parameters have default values which can be viewed using print(params.info).
"""
atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
atmosphere = shdom.Medium(atmospheric_grid)
atmosphere.add_scatterer(droplets, name='cloud')
atmosphere.add_scatterer(air, name='air')

# -----------------------------------------------
# ---------Calculate camera footprint at nadir --
# -----------------------------------------------

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
PIXEL_FOOTPRINT = 0.02 # km
Rsat = 600
fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
cny = int(np.floor(L/PIXEL_FOOTPRINT))
cnx = int(np.floor(L/PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]

# -------------------------------------------------------
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

# ------------------------------------------------------
# ------------------------------------------------------
# ------------------------------------------------------

#For radiances may need Nmu=16, Nphi=32 or more depending on the phase function. The defulte is
#Nmu=8, Nphi=16 .
#numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
                                             #split_accuracy=0.1,max_total_mb=100000.0)

numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
                                             adapt_grid_factor=5,
                                             split_accuracy=0.1,max_total_mb=300000.0)
# I ran the simulation with split_accuracy=0.00 and I got bad result. Why is that so?

scene_params = shdom.SceneParameters(
    wavelength=mie.wavelength,
    source=shdom.SolarSource(azimuth=180, zenith=157.5)
)

rte_solver = shdom.RteSolver(scene_params, numerical_params)
rte_solver.set_medium(atmosphere)
# it breaks here:
# max_total_mb memory limit exceeded with just base grid.
# solved by setting max_total_mb=100000.0

print(rte_solver.info)

"""
Solve the Radiative Transfer for the domain using SHDOM: SOLVE_RTE procedure (src/unpolarized/shdomsub1.f).
The outputs are the source function (J) and radiance field (I) which are stored in 
the shdom.RteSolver object. These are subsequently used for the rendering of an image.
"""


import pickle
if(1):
                                             rte_solver.solve(maxiter=1)
                                             # it breaks here:
                                             # failed to create intent(cache|hide)|optional array-- must have defined dimensions but got (-435764992,)

if(0):
                                             # save the solution:
                                             file = open('solution.pkl', 'wb')
                                             file.write(pickle.dumps(rte_solver, -1))
                                             file.close()    
                                             #with open('solution.pkl', 'wb') as f:
                                                                                          #pickle.dump(rte_solver, f)
if(0):
                                             # load the solution:
                                             file = open('solution.pkl', 'rb')
                                             data = file.read()
                                             file.close()
                                             rte_solver = pickle.loads(data)    
                                             #with open('solution.pkl', 'rb') as f:
                                                                                          #pickle.load(f)  
                                                                                          #print('loading the solution \n{}' .format(rte_solver.info))


"""
Render an image by integrating the incoming radiance along the projection geometry defines (pixels).
"""
projection = shdom.OrthographicProjection(
    bounding_box=droplets.grid.bounding_box, 
    x_resolution=0.02, 
    y_resolution=0.02, 
    azimuth=0.0, 
    zenith=0.0,
    altitude='TOA'
)

camera = shdom.Camera(shdom.RadianceSensor(), projection)
images = camera.render(rte_solver, n_jobs=40)
plt.imshow(images,cmap='gray')
plt.show()
# A Perspective trasnormation (pinhole camera).
projection = shdom.MultiViewProjection()

sc = 50
ny= sc*cny
nx = sc*cnx

origin = [[0.5*cnx*dx , 0.5*cny*dy , Rsat],
          [0.5*cnx*dx , 0.5*cny*dy , Rsat]]

lookat = [[0.5*cnx*dx , 0.5*cny*dy , 0],
          [0.5*cnx*dx , 0.5*cny*dy , 0]] 
  
for posind in range(len(lookat)):
                                             
                                             x, y, z = origin[posind]                                                 
                                             tmpproj = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
                                             tmpproj.look_at_transform(lookat[posind],[0,1,0])
                                             projection.add_projection(tmpproj)
                                             
                                             nx = int(nx/sc)
                                             ny = int(ny/sc)
                                             

camera = shdom.Camera(shdom.RadianceSensor(), projection)
#image = camera.render(rte_solver)
images = camera.render(rte_solver, n_jobs=40)

image_large = np.array(images[0])
image_small = np.array(images[1])
image_large = (image_large/np.max(image_large))**0.5
image_small = (image_small/np.max(image_small))**0.5

SHOWRENDERING = True

if(SHOWRENDERING):
                                             f, axarr = plt.subplots(1, 2, figsize=(20, 20))
                                             ax = axarr.ravel()
                                             
                                             ax[0].imshow(image_large,cmap='gray')
                                             ax[0].invert_xaxis() 
                                             ax[0].invert_yaxis() 
                                             ax[0].axis('off')
                                             ax[0].set_title('cloudct')
                                             
                                             ax[1].imshow(image_small,cmap='gray')
                                             ax[1].invert_xaxis() 
                                             ax[1].invert_yaxis() 
                                             ax[1].axis('off')


#file_name = 'high_angular_res'+'.mat'
file_name = 'moderate_angular_res_new_param_sa01E-1'+'.mat'
sio.savemat(file_name, {'img':images})
            
plt.show()