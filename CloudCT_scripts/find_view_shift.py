import numpy as np
import pandas as pd
import scipy.io as sio
import shdom
import glob
import time
import matplotlib.pyplot as plt
from shdom import CloudCT_setup
from tqdm import tqdm
from mayavi import mlab

from shdom.AirMSPI import AirMSPIMeasurements

mie = shdom.MiePolydisperse()
mie.read_table(file_path="../mie_tables/polydisperse/Water_660nm.scat")
surface_albedo = 0.05

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

file_name = "/home/vhold/Yael_shdom/pyshdom/yael_clouds/cloud668.txt"
start_process_time = time.time()
cloud_index = 668#file_name.split("/home/vhold/Yael_shdom/pyshdom/yael_clouds/cloud")[-1].split(".txt")[0]

droplets = shdom.MicrophysicalScatterer()        

print(cloud_index)
droplets.load_from_csv(file_name,veff=0.1)

lwc = droplets.lwc.data
veff = droplets.veff.data
reff = droplets.reff.data

new_lwc = shdom.GridData(droplets.grid,lwc)
new_veff = shdom.GridData(droplets.grid,veff)
new_reff = shdom.GridData(droplets.grid,reff)

new_droplets = shdom.MicrophysicalScatterer(new_lwc,new_reff,new_veff)
droplets = new_droplets
droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv("../ancillary_data/AFGL_summer_mid_lat.txt", comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
rayleigh = shdom.Rayleigh(wavelength=0.660)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()

atmospheric_grid = droplets.grid + air.grid
atmosphere = shdom.Medium(atmospheric_grid)
atmosphere.add_scatterer(air, name='air')
atmosphere.add_scatterer(droplets, name='cloud')

values = np.array([0.0,0.0])
atmosphere.pad_scatterer(name = 'cloud', axis=2, right = True, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=2, right = False, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=1, right = True, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=1, right = False, values=values)        
atmosphere.pad_scatterer(name = 'cloud', axis=0, right = True, values=values)
atmosphere.pad_scatterer(name = 'cloud', axis=0, right = False, values=values)
    
numerical_params = shdom.NumericalParameters(num_mu_bins=8, num_phi_bins=16, adapt_grid_factor=5,
                                             split_accuracy=0.1,
                                             max_total_mb=300000.0, num_sh_term_factor=5)
scene_params = shdom.SceneParameters(wavelength=mie.wavelength,
                                     surface=shdom.LambertianSurface(albedo=surface_albedo),
                                     source=shdom.SolarSource(azimuth=0, zenith=132.7))

rte_solver = shdom.RteSolver(scene_params, numerical_params)
rte_solver.set_medium(atmosphere)

#rte_solver.init_solution()
#rte_solver.solve(maxiter=1)

zenith_list = [70.5, 60, 45.6, 26.1, 0, 26.1, 45.6, 60, 70.5]
azimuth_list = [90, 90, 90, 90, 0, -90, -90, -90, -90]

M = AirMSPIMeasurements()
rois = 9*[[1540, 1784, 792, 1036]]
#rois = 9*[[1440, 1984, 792, 1136]]

valid_wavelength = [660]
data_dir = '../raw_data'
M.load_from_hdf(data_dir,region_of_interest=rois,valid_wavelength=valid_wavelength)
camera = M.camera

# -----------------
fig, ax = plt.subplots(1, 1)
Col = ['r','g','b','k','y','m','c','r','g','b']

p=[[-0.018,5.3],[0.07,4.34],[0.13,3.45],[0.2,2.64],[0.3,1.8],
[0.37,0.94],[0.44,0.16],[0.52,-0.63],[0.6,-1.53]]
#-----------------------
projections = M._projections
new_projections = shdom.MultiViewProjection()
for i, pro in enumerate(projections.projection_list):
    pro.x -= p[i][0]
    pro.y -= p[i][1]
    new_projections.add_projection(pro)
    
#for i,pro in enumerate(camera.projection.projection_list):
for i,pro in enumerate(new_projections.projection_list):
    
    z_c = -pro.mu
    x_c = -np.sin(np.arccos(-z_c))*np.cos(pro.phi) 
    y_c = -np.sin(np.arccos(-z_c))*np.sin(pro.phi) 
    H = 1.5
    print(i)
    inter_xs = []#np.zeros_like(x_c)
    inter_ys = []#np.zeros_like(x_c)
    inter_zs = []#np.zeros_like(x_c)
    a = 10
    ref_x = 0.0
    ref_y = 0.0
    ref_z = 0.0
    
    for ray_ind in np.arange(0,x_c.size,a):
        #print(ray_ind)
        rayPoint = pro.x[ray_ind], pro.y[ray_ind], pro.z[ray_ind]
        d = x_c[ray_ind], y_c[ray_ind], z_c[ray_ind]
        inter_x, inter_y, inter_z = LinePlaneCollision(np.array([0,0,1]), np.array([0,0,H]), np.array(d), rayPoint, 1e-6)
        inter_xs.append(inter_x)
        inter_ys.append(inter_y)
        inter_zs.append(inter_z)
    
    index_min_y = np.argmin(np.array(inter_ys))
    view_side_y = inter_ys[index_min_y]
    view_side_x =  inter_xs[index_min_y]
    #view_side_z = inter_zs.min()
    print(view_side_x,view_side_y)
    
    col = tuple(*(np.random.rand(1,3).tolist()))
    #mlab.points3d(inter_xs, inter_ys, inter_zs, 
                                #scale_factor=0.0003,color=col)
    
    #mlab.show()
    
    ax.scatter(inter_xs, inter_ys, s=5, marker='*', color = Col[i])
    #plt.show()

print(p)
ax.axis('equal')
plt.show()







