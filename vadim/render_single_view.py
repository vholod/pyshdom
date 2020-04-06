
import os 
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom


mie = shdom.MiePolydisperse()
mie.read_table(file_path='../mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
#droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
droplets.load_from_csv('../synthetic_cloud_fields/small_cloud_les/view55_small.txt', veff=0.1)

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
                                             rte_solver.solve(maxiter=100)
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
#projection = shdom.OrthographicProjection(
    #bounding_box=droplets.grid.bounding_box, 
    #x_resolution=0.02, 
    #y_resolution=0.02, 
    #azimuth=0.0, 
    #zenith=0.0,
    #altitude='TOA'
#)

# A Perspective trasnormation (pinhole camera).
projection = shdom.MultiViewProjection()
fov = 8*0.213904
ny= 10*328
nx = 10*328#2*226
#origin = [[-299907.40615228115*1e-3 , 0.0 , 593545.6813589074*1e-3],
          #[-149988.42496515365*1e-3 , 0.0 , 598386.2335485807*1e-3],
          #[-74998.55309552186*1e-3 , 0.0 , 599596.5467120232*1e-3],
          #[0.0 , 0.0 , 600000.0*1e-3],
          #[74998.55309552186*1e-3 , 0.0 , 599596.5467120232*1e-3],
          #[149988.42496515365*1e-3 , 0.0 , 598386.2335485807*1e-3],
          #[299907.40615228115*1e-3 ,
           
origin = [[0.0 , 0.0 , 100.0],
          [0.0 , 0.0 , 100.0]]

lookat = [[57*0.02 , 82*0.02 , 0],
          [57*0.02 , 82*0.02 , 0]]           

  
for posind in range(len(lookat)):
                                             
                                             x, y, z = origin[posind]                                                 
                                             tmpproj = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
                                             tmpproj.look_at_transform(lookat[posind],[0,1,0])
                                             projection.add_projection(tmpproj)
                                             
                                             nx = int(nx/10)
                                             ny = int(ny/10)
                                             

camera = shdom.Camera(shdom.RadianceSensor(), projection)
#image = camera.render(rte_solver)
images = camera.render(rte_solver, n_jobs=40)

image_large = np.array(images[0])
image_small = np.array(images[1])
image_large = (image_large/np.max(image_large))**0.5
image_small = (image_small/np.max(image_small))**0.5

f, axarr = plt.subplots(1, 2, figsize=(20, 20))
ax = axarr.ravel()

ax[0].imshow(image_large,cmap='gray')
ax[0].invert_xaxis() 
ax[0].invert_yaxis() 
ax[0].axis('off')

ax[1].imshow(image_small,cmap='gray')
ax[1].invert_xaxis() 
ax[1].invert_yaxis() 
ax[1].axis('off')


file_name = 'orto'+'.mat'
sio.savemat(file_name, {'img':images})
            
plt.show()