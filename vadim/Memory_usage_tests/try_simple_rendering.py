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
#pic mediume:
pic_index = 6

fx_list = [21,31,61,81,101,121,152]
fy_list = [21,31,61,81,101,121,252]
fz_list = [21,31,61,67,67,67,67]

origin_list =  [[0.2, 0.2, 0.91],\
                        [0.3, 0.3, 1.34333],\
                [0.6, 0.6, 2.64333],\
                [0.8, 0.8, 2.9033],\
                [1, 1, 2.90333],\
                [1.2, 1.2, 2.90333],\
                [1.5, 2.5, 2.90333]]

fov_list = [85.4187799, 105.4187799, 85.41877, 96.27356, \
                    108.59441, 118.08409, 147.86411]

lookat_list = [[0.21, 0.21, 0.56],\
                       [0.31, 0.31, 0.826667],\
                [0.61, 0.61, 1.62667],\
                [0.81, 0.81, 1.78667],\
                [1.01, 1.01, 1.78667],\
                [1.21, 1.21, 1.78667],\
                [1.52, 2.52, 1.78667]]

fx = fx_list[pic_index]
fy = fy_list[pic_index]
fz = fz_list[pic_index]

path = 'mediums/tamar_cvpr_rico{}x{}x{}.txt'.format(fx,fy,fz)
cam_nx = fx
cam_ny = fy
origin = origin_list[pic_index]
fov = fov_list[pic_index]
lookat = lookat_list[pic_index]
USE_AIR = False

# Mie scattering for water droplets
mie = shdom.MiePolydisperse()
mie.read_table(file_path='../mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv(path, veff=0.1)

droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('../../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
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
if(USE_AIR):
    atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    atmosphere.add_scatterer(air, name='air')
else:
    atmospheric_grid = droplets.grid
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    
#numerical_params = shdom.NumericalParameters()
numerical_params = shdom.NumericalParameters(num_mu_bins=16,num_phi_bins=32,adapt_grid_factor=1.1,
                                             split_accuracy=0.03,max_total_mb=100000.0,high_order_radiance=False,acceleration_flag=True)
scene_params = shdom.SceneParameters(
    wavelength=mie.wavelength,
    surface=shdom.LambertianSurface(albedo=0.009999),
    source=shdom.SolarSource(azimuth=0, zenith=180)
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

         
# render 2 views and see the camera x,y,z , angles:
for posind in range(len(lookat)):
    x, y, z = origin  

    tmpproj = shdom.PerspectiveProjection(fov, cam_nx, cam_ny, x, y, z)
    tmpproj.look_at_transform(lookat,[0,1,0])
    PHI, THETA = tmpproj.export_thete_phi()
    projection.add_projection(tmpproj)
              
    
camera = shdom.Camera(shdom.RadianceSensor(), projection)
images = camera.render(rte_solver, n_jobs=20)

# show the renderings:
f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
tookat_index = 0
for ax, image in zip(axarr, images):
    #image2show = resize_image(image=image,desired_res=[nx,ny])
    image2show = image
    ax.imshow(image2show,cmap='gray')
    ax.invert_xaxis() 
    ax.invert_yaxis() 
    ax.axis('off') 
    ax.set_title('tookat_index is {}'.format(tookat_index))
    tookat_index = tookat_index + 1
    


plt.show()

