"""
Imports necessary for this notebook
"""
import os 
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom



# Mie scattering for water droplets
mie = shdom.MiePolydisperse()
mie.read_table(file_path='../mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
droplets.add_mie(mie)

Grid_bounding_box = droplets.bounding_box
Grid_shape = droplets.grid.shape
xgrid = np.linspace(Grid_bounding_box.xmin, Grid_bounding_box.xmax,Grid_shape[0])
ygrid = np.linspace(Grid_bounding_box.ymin, Grid_bounding_box.ymax,Grid_shape[1])
zgrid = np.linspace(Grid_bounding_box.zmin, Grid_bounding_box.zmax,Grid_shape[2])   
X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
LWC_MAT = droplets.lwc.data

#scf = mlab.pipeline.scalar_field(X, Y, Z,LWC_MAT)
#figh = mlab.gcf()
##mlab.pipeline.volume(scf, figure=figh) # no working on servers
#isosurface = mlab.pipeline.iso_surface(scf, contours=[0.0001],color = (1, 1,1),opacity=0.1,transparent=True)
#mlab.pipeline.volume(isosurface, figure=figh)
#mlab.pipeline.image_plane_widget(scf,plane_orientation='x_axes',slice_index=10)
                            
#figh.scene.anti_aliasing_frames = 0
#mlab.scalarbar()
#mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 


# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
rayleigh = shdom.Rayleigh(wavelength=0.672)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()
atmospheric_grid = droplets.grid + air.grid

#AIR_MAT = air.data

#scf = mlab.pipeline.scalar_field(X, Y, Z,AIR_MAT) 
#figh = mlab.gcf()
#mlab.pipeline.image_plane_widget(scf,plane_orientation='x_axes',slice_index=10)

#mlab.scalarbar()
#mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 


#mlab.show() 
print("done")