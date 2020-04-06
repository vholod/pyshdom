import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle

def resize_image(image,desired_res=[10,10]):
    
    nx = desired_res[0]
    ny = desired_res[1]
    image_out = image
    image_out = np.zeros([nx,ny])
    image_in_size = image.shape
    sc = int(image_in_size[0]/nx)
    assert sc == int(image_in_size[1]/ny), 'resize_image function support only square pixels'
    
    for iy in range(ny):
        for ix in range(nx):
    
            image_out[ix,iy] = np.mean(image[(sc*ix):(sc*(ix+1)) , (sc*iy):(sc*(iy+1))])
            
    return image_out




# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------

# Mie scattering for water droplets
mie = shdom.MiePolydisperse()
mie.read_table(file_path='./mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)

#droplets.add_mie(mie)
droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('./ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
rayleigh = shdom.Rayleigh(wavelength=0.672)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()

# -----------------------------------------------
# ---------initilize an RteSolver object------------
# -----------------------------------------------

atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
atmosphere = shdom.Medium(atmospheric_grid)
atmosphere.add_scatterer(droplets, name='cloud')
atmosphere.add_scatterer(air, name='air')

#numerical_params = shdom.NumericalParameters()
numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
                                             split_accuracy=0.1,max_total_mb=100000000.0)
scene_params = shdom.SceneParameters(
    wavelength=mie.wavelength,
    surface=shdom.LambertianSurface(albedo=0.05),
    source=shdom.SolarSource(azimuth=45, zenith=175)
)
#azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
#zenith: Solar beam zenith angle in range (90,180]   

rte_solver = shdom.RteSolver(scene_params, numerical_params)
rte_solver.set_medium(atmosphere)


print(rte_solver.info)

  
  
# -----------------------------------------------
# ---------rte_solver.solve(maxiter=100)------------
# -----------------------------------------------
if(1):
    rte_solver.solve(maxiter=100)
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
        

# -----------------------------------------------
# ---------Orthographic Projection---------
# -----------------------------------------------
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
image = camera.render(rte_solver)

plt.imshow(image,cmap='gray')
plt.gca().invert_yaxis() 
plt.gca().invert_xaxis() 
plt.title('Orthographic Projection')
plt.colorbar()

# -----------------------------------------------
# ---------Perspective Projection--------
# -----------------------------------------------
# A Perspective trasnormation (pinhole camera).

dx = atmospheric_grid.dx
dy = atmospheric_grid.dy

nz = atmospheric_grid.nz
nx = atmospheric_grid.nx
ny = atmospheric_grid.ny

Lx = atmospheric_grid.bounding_box.xmax - atmospheric_grid.bounding_box.xmin
Ly = atmospheric_grid.bounding_box.ymax - atmospheric_grid.bounding_box.ymin
Lz = atmospheric_grid.bounding_box.zmax - atmospheric_grid.bounding_box.zmin
L = max(Lx,Ly)
dz = Lz/nz


projection = shdom.MultiViewProjection()
Rsat = 500 # km
fov = 0.54*np.rad2deg(2*np.arctan(L/(Rsat)))
sc = 10
cny= sc*ny
cnx = sc*nx

origin = [[0.5*nx*dx , 0.5*ny*dy , 500],
          [0.5*nx*dx , 0.5*ny*dy, 250],
          [0.5*nx*dx , 0.5*ny*dy, 100],
          [0.5*nx*dx , 0.5*ny*dy, 50],
          [0.5*nx*dx , 0.5*ny*dy, 10]]
          
lookat = [[0.5*nx*dx , 0.5*ny*dy , 0],
          [0.5*nx*dx , 0.5*ny*dy , 0],
          [0.5*nx*dx , 0.5*ny*dy , 0],
          [0.5*nx*dx , 0.5*ny*dy , 0],
          [0.5*nx*dx , 0.5*ny*dy , 0]]          
          
Rsat = [500,250,100,50,10]
# render 2 views and see the camera x,y,z , angles:
if(1):
    for posind in range(len(lookat)):
        x, y, z = origin[posind]  
        fov = 0.54*np.rad2deg(2*np.arctan(L/(Rsat[posind])))
        
        tmpproj = shdom.PerspectiveProjection(fov, cnx, cny, x, y, z)
        tmpproj.look_at_transform(lookat[posind],[0,1,0])
        PHI, THETA, X, Y, Z = tmpproj.export_thete_phi()
        projection.add_projection(tmpproj)
        if(1):
            fig, ax_list = plt.subplots(1, 2, figsize=(10, 10))
            ax = ax_list.ravel()
            pos = ax[0].imshow(PHI)
            ax[0].set_title('PHI {}'.format(posind))
            fig.colorbar(pos, ax=ax[0])    
            
            pos = ax[1].imshow(THETA)
            ax[1].set_title('THETA')
            fig.colorbar(pos, ax=ax[1])  
            
              
    
camera = shdom.Camera(shdom.RadianceSensor(), projection)
images = camera.render(rte_solver, n_jobs=20)

# show the renderings:
f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
tookat_index = 0
for ax, image in zip(axarr, images):
    image2show = resize_image(image=image,desired_res=[nx,ny])
    image2show = image
    ax.imshow(image2show,cmap='gray')
    ax.invert_xaxis() 
    ax.invert_yaxis() 
    ax.axis('off') 
    ax.set_title('tookat_index is {}'.format(tookat_index))
    tookat_index = tookat_index + 1
    

print("Visualize the rendering")
plt.show()

print("Finished")

"""
nice setup to work with:
Rsat = 1
print('Rsat is {} '.format(Rsat))
origin = [[2 , 0.5*ny*dy , Rsat],
          [0.5*nx*dx , 2 , Rsat]]
          
lookat = [[0.5*nx*dx , 0.5*ny*dy , Rsat+0.01],
          [0.5*nx*dx , 0.5*ny*dy , Rsat+0.01]]


fov = 45
"""