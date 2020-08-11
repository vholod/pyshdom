import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from shdom import plank
from shdom import CloudCT_setup, float_round, core




# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
#droplets.load_from_csv('../synthetic_cloud_fields/small_cloud_les/cut_from_dannys_clouds_d2.txt', veff=0.1)



# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv('../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, 100, 101))
#air_grid = shdom.Grid(z=np.linspace(0, 5, 10))


#wavelength_for_table = [0.5,0.6,0.7,0.8,1.5,1.55,1.6,1.65]
wavelength_for_table = [1.6]
sun_zenith = 155

for wavelength_for_table in [[0.5],[0.6] ,[0.7],[0.8],[1.5],[1.55],[1.6],[1.65]]:
    
    # Calculate irradiance of the spesific wavelength:
    # use plank function:
    temp = 5900 #K
    L_TOA = 6.8e-5*1e-9*plank(1e-6*np.array(wavelength_for_table),temp) # units fo W/(m^2)
    Cosain = np.cos(np.deg2rad((180-sun_zenith)))
    solar_flux = L_TOA*Cosain
    air = shdom.MultispectralScatterer()
    # Generate a Microphysical medium
    droplets = shdom.MicrophysicalScatterer()
    droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
    
    for wavelength in wavelength_for_table:  
        # Molecular Rayleigh scattering
        rayleigh = shdom.Rayleigh(wavelength)
        rayleigh.set_profile(temperature_profile.resample(air_grid))
        air.add_scatterer(rayleigh.get_scatterer())
        
        # Droplet Mie scattering
        mie = shdom.MiePolydisperse()
        table_path = '../mie_tables/polydisperse/Water_{}nm.scat'.format(int(1000*(wavelength)))
        mie.read_table(table_path)
        droplets.add_mie(mie)
        print(mie.wavelength)
        # here droplets.num_wavelengths = air.num_wavelengths = wavelength_num
        
    
    # -----------------------------------------------
    # ---------initilize an RteSolver object------------
    # -----------------------------------------------
    
    atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    atmosphere.add_scatterer(air, name='air')
    
    split_accuracy = 0.01
    rte_solvers = shdom.RteSolverArray()
    
    for wavelength in wavelength_for_table:
    
    
        numerical_params = shdom.NumericalParameters(num_mu_bins=16,num_phi_bins=32,
                                                     split_accuracy=split_accuracy,max_total_mb=500000.0)
    
        scene_params = shdom.SceneParameters(
            wavelength=wavelength,
            surface=shdom.LambertianSurface(albedo=0.05),
            source=shdom.SolarSource(azimuth=0, zenith=sun_zenith,flux=1) # flux=solar_flux
        ) 
        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)
        rte_solvers.add_solver(rte_solver)
        
    
    #azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
    #zenith: Solar beam zenith angle in range (90,180]   
    
    
    # -----------------------------------------------
    # ---------rte_solver.solve(maxiter=100)------------
    # -----------------------------------------------
    
    #rte_solver.solve(maxiter=250)
    maxiter = 100
    rte_solvers.solve(maxiter=maxiter)
    
    # rendereing:
    
    droplets_grid = droplets.grid
    dx = droplets_grid.dx
    dy = droplets_grid.dy
    
    nz = droplets_grid.nz
    nx = droplets_grid.nx
    ny = droplets_grid.ny
    
    Lx = droplets_grid.bounding_box.xmax - droplets_grid.bounding_box.xmin
    Ly = droplets_grid.bounding_box.ymax - droplets_grid.bounding_box.ymin
    Lz = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
    L = max(Lx,Ly)
    
    Lz_droplets = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
    dz = Lz_droplets/(nz-1)
    
    pixel_footprint = 0.02#km
    Rsat = 500
    simple_sensor = shdom.SensorFPA(PIXEL_SIZE = 1,CHeight = 100, CWidth = 100)
    simple_lens = shdom.LensSimple(FOCAL_LENGTH = 1 , DIAMETER = 1)
    imager = shdom.Imager(sensor=simple_sensor,lens=simple_lens)
    # you must define the spectrum for each defined imager:
    imager.set_scene_spectrum_in_microns([wavelength_for_table[0],wavelength_for_table[0]])
    imager.set_Imager_altitude(H=Rsat)  
    imager.set_pixel_footprint(pixel_footprint)
    
    
    #USED FOV, RESOLUTION and SAT_LOOKATS:
    # cny x cnx is the camera resolution in pixels
    fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
    cny = int(np.floor(L/pixel_footprint))
    cnx = int(np.floor(L/pixel_footprint))
    
    
    CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]
    
    # Somtimes it is more convinent to use wide fov to see the whole cloud
    # from all the view points. so the FOV is aslo tuned:
    IFTUNE_CAM = True
    # --- TUNE FOV, CNY,CNX:
    if(IFTUNE_CAM):
        L = 1.5*L
        fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
        cny = int(np.floor(L/pixel_footprint))
        cnx = int(np.floor(L/pixel_footprint))
    
    imager.update_sensor_size_with_number_of_pixels(cnx, cny)   
    
    # not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
    # tuning is applied by the variavle LOOKAT.
    LOOKAT = CENTER_OF_MEDIUM_BOTTOM
    if(IFTUNE_CAM):
        LOOKAT[2] = 0.68*nx*dz # tuning. if IFTUNE_CAM = False, just lookat the bottom
            
    SAT_LOOKATS = np.array(10*LOOKAT).reshape(-1,3)# currently, all satellites lookat the same point.
        
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
    x_min = droplets_grid.bounding_box.xmin
    x_max = droplets_grid.bounding_box.xmax
    
    y_min = droplets_grid.bounding_box.ymin
    y_max = droplets_grid.bounding_box.ymax
    
    z_min = droplets_grid.bounding_box.zmin
    z_max = droplets_grid.bounding_box.zmax 
    print("xmin = {}, ymin = {},zmin ={}".format(x_min,y_min,z_min))
    print("xmax = {}, ymax = {},zmax ={}".format(x_max,y_max,z_max))
    
    print(20*"-")
    print(20*"-")
    print(20*"-")
    
    # create CloudCT setups:
    projections, near_nadir_view_index = CloudCT_setup.Create(\
        SATS_NUMBER = 10, ORBIT_ALTITUDE = Rsat, \
        SAT_LOOKATS = SAT_LOOKATS, \
        Imager_config = 10*[True] ,imager = imager, VISSETUP = False)
        
       
    sensor=shdom.RadianceSensor()  
    camera = shdom.Camera(sensor, projections)
    radiance_images = camera.render(rte_solvers, n_jobs=20)  
    
    radiance_images = [i*solar_flux for i in radiance_images]
    
    fig = plt.figure(figsize=(20, 20))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2,5),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )  
    
    # show all
    images_array = np.array(radiance_images)
    MAXI = images_array.max()
    for projection_index, (ax, name) in enumerate(zip(grid, projections.projection_names)):
        image = radiance_images[projection_index]
        
        ax.set_axis_off()
        im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
        ax.set_title("{}".format(name))
    
    # super title:    
    
    # Since currently the view per same imager have identicale nx and ny:
    nx, ny =  image.shape
    title = "$\lambda$={} micron".format(wavelength_for_table[0])
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle(title, size=16,y=0.95)
    
          
    import scipy.io as sio
            
    file_name = 'lambda_{}nm_cloud'.format(int(1000*wavelength_for_table[0]))+'.mat'
    sio.savemat(file_name, {'img':radiance_images})
    
    print("Visualize the rendering")
    #plt.show()
    
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # ---------------- render no clouds at all -----------------:
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    #air = shdom.MultispectralScatterer()
    ## Generate a Microphysical medium
    #droplets = shdom.MicrophysicalScatterer()
    #droplets.load_from_csv('../CloudCT_notebooks/ZeroAtm40x40x40.txt', veff=0.1)
    ##droplets.load_from_csv('../synthetic_cloud_fields/small_cloud_les/cut_from_dannys_clouds_d2.txt', veff=0.1)
    
    
    #for wavelength in wavelength_for_table:  
        ## Molecular Rayleigh scattering
        #rayleigh = shdom.Rayleigh(wavelength)
        #rayleigh.set_profile(temperature_profile.resample(air_grid))
        #air.add_scatterer(rayleigh.get_scatterer())
        
        ## Droplet Mie scattering
        #mie = shdom.MiePolydisperse()
        #table_path = '../mie_tables/polydisperse/Water_{}nm.scat'.format(int(1000*(wavelength)))
        #mie.read_table(table_path)
        #droplets.add_mie(mie)
        #print(mie.wavelength)
        ## here droplets.num_wavelengths = air.num_wavelengths = wavelength_num
        
    
    ## -----------------------------------------------
    ## ---------initilize an RteSolver object------------
    ## -----------------------------------------------
    
    #atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
    #atmosphere = shdom.Medium(atmospheric_grid)
    #atmosphere.add_scatterer(droplets, name='cloud')
    #atmosphere.add_scatterer(air, name='air')
    
    
    for wavelength in wavelength_for_table:  
        # Molecular Rayleigh scattering
        
        # Droplet Mie scattering
        mie = shdom.MiePolydisperse()
        table_path = '../mie_tables/polydisperse/Water_{}nm.scat'.format(int(1000*(wavelength)))
        mie.read_table(table_path)
        droplets.add_mie(mie)
        print(mie.wavelength)
        # here droplets.num_wavelengths = air.num_wavelengths = wavelength_num
        
    
    # -----------------------------------------------
    # ---------initilize an RteSolver object------------
    # -----------------------------------------------
    atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    
    # -----------------------------------------------
    # ---------initilize an RteSolver object------------
    # -----------------------------------------------
    
    
    
    split_accuracy = 0.01
    rte_solvers = shdom.RteSolverArray()
    
    for wavelength in wavelength_for_table:
    
    
        numerical_params = shdom.NumericalParameters(num_mu_bins=16,num_phi_bins=32,
                                                     split_accuracy=split_accuracy,max_total_mb=500000.0)
    
        scene_params = shdom.SceneParameters(
            wavelength=wavelength,
            surface=shdom.LambertianSurface(albedo=0.05),
            source=shdom.SolarSource(azimuth=0, zenith=sun_zenith,flux=1) # flux=solar_flux
        ) 
        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)
        rte_solvers.add_solver(rte_solver)
        
    
    
    # -----------------------------------------------
    # ---------rte_solver.solve(maxiter=100)------------
    # -----------------------------------------------
    
    #rte_solver.solve(maxiter=250)
    rte_solvers.solve(maxiter=maxiter)
    
    radiance_images_nc = camera.render(rte_solvers, n_jobs=20)  
    radiance_images_nc = [i*solar_flux for i in radiance_images_nc]
    
    fig = plt.figure(figsize=(20, 20))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2,5),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )  
    
    # show all
    images_array = np.array(radiance_images_nc)
    MAXI = images_array.max()
    
    for projection_index, (ax, name) in enumerate(zip(grid, projections.projection_names)):
        image = radiance_images_nc[projection_index]
        
        ax.set_axis_off()
        im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
        ax.set_title("{}".format(name))
    
    # super title:    
    
    # Since currently the view per same imager have identicale nx and ny:
    nx, ny =  image.shape
    title = "no clouds $\lambda$={} micron".format(wavelength_for_table[0])
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle(title, size=16,y=0.95)
    
          
            
    file_name = 'lambda_{}nm_no_cloud'.format(int(1000*wavelength_for_table[0]))+'.mat'
    sio.savemat(file_name, {'img':radiance_images_nc})
    
    
    # -----------------------------------------------
    # ---------calc differences          ------------
    # -----------------------------------------------
    diff_images = [i-j for i,j in zip(radiance_images, radiance_images_nc)]
    file_name = 'lambda_{}nm_diff'.format(int(1000*wavelength_for_table[0]))+'.mat'
    sio.savemat(file_name, {'img':diff_images})
    
    fig = plt.figure(figsize=(20, 20))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2,5),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )  
    
    # show all
    images_array = np.array(diff_images)
    MAXI = images_array.max()
    
    for projection_index, (ax, name) in enumerate(zip(grid, projections.projection_names)):
        image = diff_images[projection_index]
        
        ax.set_axis_off()
        im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
        ax.set_title("{}".format(name))
    
    # super title:    
    
    # Since currently the view per same imager have identicale nx and ny:
    nx, ny =  image.shape
    title = "diffrences $\lambda$={} micron".format(wavelength_for_table[0])
    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    fig.suptitle(title, size=16,y=0.95)
    
    
    

print("Visualize the rendering")
plt.show()

print("Finished")


