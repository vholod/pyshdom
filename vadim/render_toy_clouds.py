
import os 
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
from math import log, exp, tan, atan, pi, ceil, modf


mie = shdom.MiePolydisperse()
mie.read_table(file_path='../mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
#droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
droplets.load_from_csv('../vadim/Mediume_Samples/long_blob_sample.txt', veff=0.1)
droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
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

#For radiances may need Nmu=16, Nphi=32 or more depending on the phase function. The defulte is
#Nmu=8, Nphi=16 .
#numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
    #split_accuracy=0.1,max_total_mb=100000.0)

numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
                                             split_accuracy=0.1,max_total_mb=100000000.0)
scene_params = shdom.SceneParameters(
    wavelength=mie.wavelength,
    source=shdom.SolarSource(azimuth=0, zenith=180)
)
"""azimuth: float,
    Solar beam azimuthal angle (photon direction); specified in degrees but immediately converted to radians for use in code. 
    0 is beam going in positive X direction (North), 90 is positive Y (East).
zenith: 
    Solar beam zenith angle in range (90,180]; Specified in degrees but immediately converted to the cosine of the angle (mu).
    This angle represents the direction of travel of the solar beam, so is forced to be negative although it can be specified positive.
"""


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
rte_solver.solve(maxiter=1)
# it breaks here:
# failed to create intent(cache|hide)|optional array-- must have defined dimensions but got (-435764992,)

"""
Render an image by integrating the incoming radiance along the projection geometry defines (pixels).
"""
#projection = shdom.OrthographicProjection(
    #bounding_box=droplets.grid.bounding_box, 
    #x_resolution=0.02, 
    #y_resolution=0.02, 
    #azimuth=0.0, 
    #zenith=0.0,
    #altitude='TOA'
#)

if(0):
    projection = shdom.MultiViewProjection()
    projection.add_projection(shdom.OrthographicProjection(
        bounding_box=droplets.grid.bounding_box, 
        x_resolution=0.002, 
        y_resolution=0.002, 
        azimuth=0.0, 
        zenith=0.0,
        altitude='TOA'
    ))
    
    camera = shdom.Camera(shdom.RadianceSensor(), projection)
    image = camera.render(rte_solver, n_jobs=3)    
    
    plt.imshow(image[0],cmap='gray')
    plt.show()
    

if(1):
    # A Perspective trasnormation (pinhole camera).
    projection = shdom.MultiViewProjection()
    fov = 4*np.rad2deg(2*atan(0.5/600))
    ny= 4*50
    nx = 4*50
    
    
    
    origin = [[0.0 , 0.0 , 600],
              [100 , 0.0 , 600],
              [200 , 0.0 , 600]]
    
    lookat = [[25*0.02 , 25*0.02 , 0],
              [25*0.02 , 25*0.02 , 0],
              [0*0.02 , 0*0.02 , 0]]
    
    
    for posind in range(len(lookat)):
        x, y, z = origin[posind]                                                 
        tmpproj = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
        tmpproj.look_at_transform(lookat[posind],[0,1,0])
        PHI, THETA = tmpproj.export_thete_phi()
        if(1):
            fig, ax_list = plt.subplots(1, 2, figsize=(10, 20))
            ax = ax_list.ravel()
            pos = ax[0].imshow(PHI)
            ax[0].set_title('PHI {}'.format(posind))
            fig.colorbar(pos, ax=ax[0])    
            
            pos = ax[1].imshow(THETA)
            ax[1].set_title('THETA')
            fig.colorbar(pos, ax=ax[1])  
                
        projection.add_projection(tmpproj)
    
    camera = shdom.Camera(shdom.RadianceSensor(), projection)
    images = camera.render(rte_solver, n_jobs=3)
    
    f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
    for ax, image in zip(axarr, images):
        ax.imshow(image,cmap='gray')
        ax.invert_xaxis() 
        ax.invert_yaxis() 
        ax.axis('off')
    
    # gamma corection:
    gamma_images = np.array(images) # np.array(images).shape is (N,H,W)
    gamma_images = (gamma_images/np.max(gamma_images))**0.5
    images = []
    for i in range(gamma_images.shape[0]):
        images.append(gamma_images[i])
    
    
    f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
    for ax, image in zip(axarr, images):
        ax.imshow(image,cmap='gray',vmin=0.0, vmax=1.0)
        ax.invert_xaxis() 
        ax.invert_yaxis() 
        ax.axis('off')






file_name = 'orto'+'.mat'
sio.savemat(file_name, {'img':images})

plt.show()