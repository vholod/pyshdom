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

# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS BELOW----------------------
# -----------------------------------------------------------------
def plank(llambda,T):
    h = 6.62607004e-34 # Planck constant
    c = 3.0e8
    k = 1.38064852e-23 # Boltzmann constant
    # https://en.wikipedia.org/wiki/Planck%27s_law
    a = 2.0*h*(c**2)
    b = (h*c)/(llambda*k*T)
    spectral_radiance = a/ ( (llambda**5) * (np.exp(b) - 1.0) )
    return spectral_radiance

def CALC_MIE_TABLES(where_to_check_path = './mie_tables/polydisperse',wavelength_micron=None):
    """
    Check if mie tables exist, if not creat them, if yes skip it is long process.
    table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
    Parameters
    ----------
    where_to_check_path: string, a path to chech (and or create) the mie table.
    
    wavelength_micron: float, the wavelength in microns
    """    
    
    # YOU MAY TUNE THE MIE TABLE PARAMETERS HERE:
    start_reff = 1 # Starting effective radius [Micron]
    end_reff = 25.0
    num_reff = int((end_reff-start_reff)/0.25 + 1)
    start_veff = 0.05
    end_veff = 0.4
    num_veff = int((end_veff-start_veff)/0.003 + 1)
    radius_cutoff = 65.0 # The cutoff radius for the pdf averaging [Micron]
    
    
    # --------------------------------------------------------------
    assert wavelength_micron is not None, "You must provied the wavelength"

        
    if(np.isscalar(wavelength_micron)):
        wavelength_list = [wavelength_micron]
    else:
        if(not isinstance(wavelength_micron, list)):
            wavelength_list = wavelength_micron.tolist()
        else:
            wavelength_list = wavelength_micron
    
    
    
    if(os.path.exists(where_to_check_path)):
        import glob        
        mie_tables_paths = sorted(glob.glob(where_to_check_path + '/Water_*.scat'))
        mie_tables_names = [os.path.split(i)[-1] for i in mie_tables_paths]
        # extract the wavelength:
        import re
        exist_wavelengths_list = [re.findall('Water_(\d*)nm.scat', i)[0] for i in mie_tables_names]# wavelength that already has mie table.
        exist_wavelengths_list = [int(i) for i in exist_wavelengths_list]# convert to integer 
        # the wavelength_list is in microne. the exist_wavelengths_list is in nm
        wavelength_list_final = []
        for _wavelength_ in wavelength_list:
            print('Check if wavelength {}um has already a table.'.format(_wavelength_))
            if(1e3*_wavelength_ in exist_wavelengths_list):
                print('Does exist, skip its creation\n')
            else:
                print('Does not exist, it will be created\n')
                wavelength_list_final.append(_wavelength_)
                            
    
    if(not (wavelength_list_final==[])):
        
        # importing functools for reduce() 
        import functools  
        # importing operator for operator functions 
        import operator 
        
        wavelength_string = functools.reduce(operator.add,[str(j)+" " for j in wavelength_list_final]).rstrip()
        wavelength_arg = ' --wavelength '+wavelength_string
        print(wavelength_arg)
        """
        wavelength:
        Wavelengths [Micron] for which to compute the polarized mie table' \
        'The output file name is formated as: Water_<wavelength[nm]>nm.scat<pol>
        """
        
        #------------
        SCRIPTS_PATH = '../scripts'
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'generate_mie_tables.py')+\
                ' --start_reff '+ str(start_reff) +\
                ' --end_reff '+ str(end_reff) +\
                ' --num_reff '+ str(num_reff) +\
                ' --start_veff '+ str(start_veff) +\
                ' --end_veff '+ str(end_veff) +\
                ' --num_veff '+ str(num_veff) +\
                ' --radius_cutoff '+ str(radius_cutoff) +\
                ' --poly_dir ' + where_to_check_path +\
                ' --mono_dir ' + where_to_check_path.replace('polydisperse','monodisperse') +\
                wavelength_arg
        
        Generate_Mie_scat = subprocess.call( cmd, shell=True)
        
    print('Mie table is calculated.')

    print(20*'-')
    print(20*'-')
    
    mie_base_path = []
    for _wavelength_ in wavelength_list:
        mie_base_path.append(where_to_check_path + '/Water_{}nm.scat'. \
        format(int(1e3 * _wavelength_)))
        
    
    return mie_base_path


    
# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS ABOVE----------------------
# -----------------------------------------------------------------

# visualization params:
VISSETUP = False
scale = 500
axisWidth = 0.02
axisLenght = 5000  

# sat number:
#SATS_NUMBER = 10
n_jobs = 30


# orbit altitude:
Rsat = 500 # km
wavelength_micron = 0.672  #0.672 , 1.6
sun_azimuth = 45
sun_zenith = 150
SATS_NUMBER_SETUP = 10 # satellites number to build the setup, for the inverse, we can use less satellites.
SATS_NUMBER_INVERSE = SATS_NUMBER_SETUP#10 # how much satelliets will be used for the inverse.

"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""
where_to_check_path = './mie_tables/polydisperse'
mie_base_path = CALC_MIE_TABLES(where_to_check_path,\
                wavelength_micron)
mie_base_path = mie_base_path[0]
# for example, mie_base_path = './mie_tables/polydisperse/Water_672nm.scat'



# forward_dir, where to save evrerything that is related to forward model:
forward_dir = './experiments/unity_flux_active_sats_{}_LES_cloud_field_rico_Water_{}nm/monochromatic'.format(SATS_NUMBER_SETUP,int(1e3*wavelength_micron))
# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name_base = 'active_sats_{}_easiest_rico32x37x26_{}'.format(SATS_NUMBER_SETUP,int(1e3*wavelength_micron))
# Write intermediate TensorBoardX results into log_name.
# The provided string is added as a comment to the specific run.

air_path = './ancillary_data/AFGL_summer_mid_lat.txt' # Path to csv file which contains temperature measurements
air_num_points = 20 # Number of altitude grid points for the air volume
air_max_alt = 5 # in km ,Maximum altitude for the air volume

# --------------------------------------------------------

DOFORWARD = True
DOINVERSE = True



# ---------------------------------------------------------------
# -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
# ---------------------------------------------------------------

# Mie scattering for water droplets
mie = shdom.MiePolydisperse()
mie.read_table(file_path = mie_base_path)

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)

droplets.add_mie(mie)

# Rayleigh scattering for air molecules up to 20 km
df = pd.read_csv(air_path, comment='#', sep=' ')
altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
air_grid = shdom.Grid(z=np.linspace(0, air_max_alt ,air_num_points))
rayleigh = shdom.Rayleigh(wavelength=wavelength_micron)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()

atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
atmosphere = shdom.Medium(atmospheric_grid)
atmosphere.add_scatterer(droplets, name='cloud')
atmosphere.add_scatterer(air, name='air')

# Calculate irradiance of the spesific wavelength:
# use plank function:
temp = 5900 #K
L_TOA = 6.8e-5*1e-9*plank(1e-6*wavelength_micron,temp) # units fo W/(m^2)
Cosain = np.cos(np.deg2rad((180-sun_zenith)))
solar_flux_scale = L_TOA*Cosain # the forward simulation will run with unity flux,
# this is a scale that we should consider to apply on the output images.
# Rigth now, lets skip it.

solar_flux = 1 # unity flux
split_accuracy = 0.1 # 0.1 nice results, For the rico cloud i didn't see that decreasing the split accuracy improves the rec.
# so fo the rico loud Let's use split_accuracy = 0.1

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
fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
cny = int(np.floor(L/PIXEL_FOOTPRINT))
cnx = int(np.floor(L/PIXEL_FOOTPRINT))

CENTER_OF_MEDIUM_BOTTOM = [0.5*nx*dx , 0.5*ny*dy , 0]

# Somtimes it is more convinent to use wide fov to see the whole cloud
# from all the view points. so the FOV is aslo tuned:
IFTUNE_CAM = True
# --- TUNE FOV, CNY,CNX:
if(IFTUNE_CAM):
    L = 1.5*L
    fov = 2*np.rad2deg(np.arctan(0.5*L/(Rsat)))
    cny = int(np.floor(L/PIXEL_FOOTPRINT))
    cnx = int(np.floor(L/PIXEL_FOOTPRINT))    

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
    
VISSETUP = False
# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variavle LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
if(IFTUNE_CAM):
    LOOKAT[2] = 0.68*nx*dz # tuning. if IFTUNE_CAM = False, just lookat the bottom
    
    
SAT_LOOKATS = np.array(SATS_NUMBER_SETUP*LOOKAT).reshape(-1,3)# currently, all satellites lookat the same point.   

# create CloudCT setup:
CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(\
    SATS_NUMBER = SATS_NUMBER_SETUP, ORBIT_ALTITUDE = Rsat, \
    CAM_FOV = fov, CAM_RES = (cnx,cny), SAT_LOOKATS = SAT_LOOKATS, VISSETUP = VISSETUP)
""" 
How to randomly choose N views from the total views:
for i in range(10):
    NEW = CloudCT_VIEWS.Random_choose_N(3)
    NEW.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
    figh = mlab.gcf()
    mlab.orientation_axes(figure=figh)    
    mlab.show()
""" 

"""
Solve the forward model:
"""
if(DOFORWARD):            
    # -----------------------------------------------
    # -----------------------------------------------
    # -----------------------------------------------
        
    # -----------------------------------------------
    # ---------initilize an RteSolver object------------
    # -----------------------------------------------
        
        
        
    #numerical_params = shdom.NumericalParameters()
    
    numerical_params = shdom.NumericalParameters(num_mu_bins=8,num_phi_bins=16,
                                                 split_accuracy=split_accuracy,max_total_mb=300000.0)
    
    
    scene_params = shdom.SceneParameters(
        wavelength=mie.wavelength,
        surface=shdom.LambertianSurface(albedo=0.05),
        source=shdom.SolarSource(azimuth = sun_azimuth,
                                 zenith = sun_zenith,flux = solar_flux)
    )
    #azimuth: 0 is beam going in positive X direction (North), 90 is positive Y (East).
    #zenith: Solar beam zenith angle in range (90,180]   
    
    rte_solver = shdom.RteSolver(scene_params, numerical_params)
    rte_solver.set_medium(atmosphere)
    print(rte_solver.info)
    
    print(20*"-")
    print(20*"-")
    print(20*"-")  
    
    # -----------------------------------------------
    # ---------RTE SOLVE ----------------------------
    # -----------------------------------------------
    
    rte_solver.solve(maxiter=100)
        
    
    # -----------------------------------------------
    # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
    # -----------------------------------------------
    """
    Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
    """
    camera = shdom.Camera(shdom.RadianceSensor(), CloudCT_VIEWS)
    # render all:
    images = camera.render(rte_solver, n_jobs=n_jobs)
    
    projection_names = CloudCT_VIEWS.projection_names
    # calculate images maximum:
    images_array = np.array(images)
    MAXI = images_array.max()
    
    
    SEE_IMAGES = True
    if(SEE_IMAGES):
    # nice plot alternative:
        fig = plt.figure(figsize=(20, 20))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, int(round(np.ceil(SATS_NUMBER_SETUP/2)))),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1
                        )  
        
        for ax, image, name in zip(grid, images, projection_names):
            ax.set_axis_off()
            im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
            ax.set_title("{}".format(name))
         
        title = "$\lambda$={}nm , $\Phi$={:.2f} , $L$={:.2f}".format(int(1e3*wavelength_micron),sun_zenith,solar_flux)
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        fig.suptitle(title, size=16,y=0.95) 
            
        plt.show()
    
    
    # -----------------------------------------------
    # ---------SAVE EVERYTHING FOR THIS SETUP -------
    # -----------------------------------------------
    """
    It is bad that:
    type(camera.projection)
    <class 'CloudCT_setup.SpaceMultiView'>
    """
    measurements = shdom.Measurements(camera, images=images, wavelength=rte_solver.wavelength)
    medium = atmosphere
    shdom.save_forward_model(forward_dir, medium, rte_solver, measurements)


# -----------------------------------------------
# ---------SOLVE INVERSE ------------------------
# -----------------------------------------------
"""
Solve inverse problem:
"""
if(DOINVERSE):
    SEE_SETUP = False
    SEE_IMAGES = True
    
    MICROPHYSICS = True
    
    # load the measurments to see the rendered images:
    medium, solver, measurements = shdom.load_forward_model(forward_dir)
    # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    USED_CAMERA = measurements.camera
    RENDERED_IMAGES = measurements.images    
    THIS_MULTI_VIEW_SETUP = USED_CAMERA.projection
    
    # Get optical medium ground-truth
    scatterer_name='cloud'
    ground_truth = medium.get_scatterer(scatterer_name)
    if isinstance(ground_truth, shdom.MicrophysicalScatterer):
        ground_truth = ground_truth.get_optical_scatterer(measurements.wavelength)
    
    """
    IMPORTANT:
    Later on, We need to add niose to the measurements
    we can do it here, like: 
    measurements = noise.apply(measurements)
    """      
    
    # show images:
    if(SEE_IMAGES):
        
        """
        It is a good place to pick the radiance_threshold = Threshold for the radiance to create a cloud mask.
        So here, we seee the original images and below we will see the images after the radiance_threshold: 
        """
        # --------------------------------------------------
        #  ----------------original:------------------------
        # --------------------------------------------------
        RENDERED_IMAGES = measurements.images
        projection_names = CloudCT_VIEWS.projection_names
        # calculate images maximum:
        images_array = np.array(RENDERED_IMAGES)
        MAXI = images_array.max()
        
        # nice plot alternative:
        fig = plt.figure(figsize=(20, 20))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, int(round(np.ceil(SATS_NUMBER_SETUP/2)))),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1
                        )  
        
        for ax, image, name in zip(grid, RENDERED_IMAGES, projection_names):
            ax.set_axis_off()
            im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
            ax.set_title("{}".format(name))
         
        title = "$\lambda$={}nm , $\Phi$={:.2f} , $L$={:.2f}".format(int(1e3*wavelength_micron),sun_zenith,solar_flux)
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        fig.suptitle(title, size=16,y=0.95)             
            
            
        # --------------------------------------------------
        #  ----------------try radiance_threshold value:----
        # --------------------------------------------------
        #radiance_threshold = 0.04 # Threshold for the radiance to create a cloud mask.
        radiance_threshold = 0.025 # Threshold for the radiance to create a cloud mask.
        # Threshold is either a scalar or a list of length of measurements.
        # radiance_threshold = 0.02 good for SWIR and rico cloud.
        # radiance_threshold = 0.025 good for 0.672 um and rico cloud.
        
        
        fig = plt.figure(figsize=(20, 20))
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2, int(round(np.ceil(SATS_NUMBER_SETUP/2)))),
                        axes_pad=0.3,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1
                        )  
        
        for ax, image, name in zip(grid, RENDERED_IMAGES, projection_names):
            ax.set_axis_off()
            tmp_image = image
            tmp_image[tmp_image<=radiance_threshold] = 0
            im = ax.imshow(tmp_image,cmap='gray',vmin=0, vmax=MAXI)
            ax.set_title("{}".format(name))
         
        title = "After radiance_threshold of {}\n$\lambda$={}nm , $\Phi$={:.2f} , $L$={:.2f}".format(radiance_threshold,int(1e3*wavelength_micron),sun_zenith,solar_flux)
        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        fig.suptitle(title, size=16,y=0.95)                
        
        
    """
    IF we want less than SATS_NUMBER_SETUP (usually 10) satellites, this is a good place to 
    choose the SATS_NUMBER_INVERSE images for the inverse.
    The measurments object is also updated here to include less satellites.
    """
    if(SATS_NUMBER_INVERSE<SATS_NUMBER_SETUP):
        if((SATS_NUMBER_INVERSE%2)==0):
            new_image_indexes = np.arange(1+near_nadir_view_index-0.5*SATS_NUMBER_INVERSE,1+near_nadir_view_index+0.5*SATS_NUMBER_INVERSE)
            #cut the edges.
        else:
            new_image_indexes = np.arange(0.5+near_nadir_view_index-0.5*SATS_NUMBER_INVERSE,0.5+near_nadir_view_index+0.5*SATS_NUMBER_INVERSE)
            # preserve near_nadir_view_index in the middle.
        new_image_indexes = [int(i) for i in new_image_indexes]
        assert len(new_image_indexes)== SATS_NUMBER_INVERSE, "Error in choosing the number of satellites for the inverse"
    
        # update measurements that will be used in the inverse:
        measurements = shdom.Measurements(USED_CAMERA,
                                          images=[RENDERED_IMAGES[i] for i in new_image_indexes],
                                          wavelength=solver.wavelength)
        
        # show images that will be used in the inverse:
        if(SEE_IMAGES):        

            new_projection_names = [projection_names[i] for i in new_image_indexes]
            RENDERED_IMAGES = measurements.images
            fig = plt.figure(figsize=(20, 20))
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(2, int(round(np.ceil(SATS_NUMBER_SETUP/2)))),
                            axes_pad=0.3,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1
                            )  
            zero_image = np.zeros_like(RENDERED_IMAGES[0])
            for ax, name in zip(grid, projection_names):
                if(name in new_projection_names):
                    image = RENDERED_IMAGES[new_projection_names.index(name)]
                    im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
                else:
                    ax.imshow(zero_image,cmap='gray',vmin=0, vmax=MAXI)
                    
                ax.set_axis_off()
                ax.set_title("{}".format(name))
             
            title = "{} satellites were chosen for the inverse.\n$\lambda$={}nm , $\Phi$={:.2f} , $L$={:.2f}".format(SATS_NUMBER_INVERSE,int(1e3*wavelength_micron),sun_zenith,solar_flux)
            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[0].colorbar(im)
            fig.suptitle(title, size=16,y=0.95)                
        
            
            
            plt.show()    
    
    # show the mutli view setup if you want.
    if(SEE_SETUP):
        THIS_MULTI_VIEW_SETUP.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)    
        mlab.show()    
        
    
    # ----------------------------------------------------------
    
    """
    Work with the optimization:
    """
    
    if not MICROPHYSICS:
    
        # -----------------------------------------------
        # ---------SOLVE for extinction only  -----------
        # -----------------------------------------------    
        """
        Estimate extinction with (what is known?):
        1. ground truth phase function (use_forward_phase, with mie_base_path)
        2. grid (use_forward_grid)
        3. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
        4. known albedo (use_forward_albedo)
        5. rayleigh scattering (add_rayleigh)
        
        """
        log_name = "extinction_only_"+log_name_base
        # -----------------------------------------------
        # ---------------- inverse parameters:-----------
        # -----------------------------------------------
        
        stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                
        # Initialization and Medium parameters:
        extinction = 0.01 # init extinction of the generator
        # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
        init = 'Homogeneous'
        # The mie_base_path is defined at the begining of this script.
        INIT_USE = ' --init '+ init
        add_rayleigh = True
        use_forward_mask = False#True
        use_forward_grid = True
        use_forward_albedo = True
        use_forward_phase = True
        if_save_gt_and_carver_masks = True
        if_save_final3d = True
        
        GT_USE = ''
        GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
        GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
        GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
        GT_USE = GT_USE + ' --use_forward_albedo' if use_forward_albedo else GT_USE
        GT_USE = GT_USE + ' --use_forward_phase' if use_forward_phase else GT_USE
        GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
        GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
        
        # (use_forward_mask, use_forward_grid, use_forward_albedo, use_forward_phase):
        # Use the ground-truth things. This is an inverse crime which is 
        # usefull for debugging/development.    
        
            
        # The log_name defined above
        # Write intermediate TensorBoardX results into log_name.
        # The provided string is added as a comment to the specific run.
        
        # note that currently, the invers_dir = forward_dir.
        # In forward_dir, the forward modeling parameters are be saved.
        # If invers_dir = forward_dir:
        # The invers_dir directory will be used to save the optimization results and progress.
        
        # optimization:
        globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
        maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
        maxls = 30 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
        disp = True # Display optimization progression.
        
        gtol = 1e-16 # Stop criteria for the maximum projected gradient.
        # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        ftol = 1e-16 # Stop criteria for the relative change in loss function.
        # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
        loss_type = 'l2'
        # Different loss functions for optimization. Currently only l2 is supported.
        
        #-------------------------------------------------
        # ---- start working with the above parameters:---
        #-------------------------------------------------
        OTHER_PARAMS = ' --input_dir ' + forward_dir + \
            ' --extinction ' + str(extinction) +\
            ' --log ' + log_name +\
            ' --air_path ' + air_path +\
            ' --n_jobs '+ str(n_jobs)+\
            ' --loss_type '+ str(loss_type)+\
            ' --maxls '+ str(maxls)+\
            ' --maxiter '+ str(maxiter)+\
            ' --radiance_threshold '+ str(radiance_threshold)
        
        OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
        
        # We have: ground_truth, rte_solver, measurements.
        SCRIPTS_PATH = '../scripts'
        cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_extinction_lbfgs.py')+\
            OTHER_PARAMS + GT_USE + INIT_USE
        
        Optimaize1 = subprocess.call( cmd, shell=True)
    
    
    else: # if we here, we rec. MICROPHYSICS, but which?
        REC_only_lwc = False
        REC_only_reff = False
        REC_only_veff = False
        REC_all = True
        if(REC_only_lwc):
            
            # -----------------------------------------------
            # ---------SOLVE for lwc only  ------------------
            # -----------------------------------------------    
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and variance, Hence,
               the albedo should be known.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "lwc_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            lwc = 0.01 # init lwc of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = False#True
            use_forward_grid = True
            use_forward_reff = True
            use_forward_veff = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_reff' if use_forward_reff else GT_USE
            GT_USE = GT_USE + ' --use_forward_veff' if use_forward_veff else GT_USE                
            GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
            GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
            
            # optimization:
            globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
            maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            maxls = 30 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            disp = True # Display optimization progression.
            
            gtol = 1e-16 # Stop criteria for the maximum projected gradient.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            ftol = 1e-16 # Stop criteria for the relative change in loss function.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            loss_type = 'l2'
            # Different loss functions for optimization. Currently only l2 is supported.
            
            #-------------------------------------------------
            # ---- start working with the above parameters:---
            #-------------------------------------------------
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --lwc ' + str(lwc) +\
                ' --log ' + log_name +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)            
        
        
        if(REC_only_reff):
            
            # -----------------------------------------------
            # ---------SOLVE for reff only  ------------------
            # -----------------------------------------------    
            """
            Estimate reff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective variance and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "reff_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            reff = 15 # init reff of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = False#True
            use_forward_grid = True
            use_forward_lwc = True
            use_forward_veff = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_lwc' if use_forward_lwc else GT_USE
            GT_USE = GT_USE + ' --use_forward_veff' if use_forward_veff else GT_USE                
            GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
            GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
            

            # optimization:
            globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
            maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            maxls = 30 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            disp = True # Display optimization progression.
            
            gtol = 1e-16 # Stop criteria for the maximum projected gradient.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            ftol = 1e-16 # Stop criteria for the relative change in loss function.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            loss_type = 'l2'
            # Different loss functions for optimization. Currently only l2 is supported.
            
            #-------------------------------------------------
            # ---- start working with the above parameters:---
            #-------------------------------------------------
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --reff ' + str(reff) +\
                ' --log ' + log_name +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)  
            
        if(REC_only_veff):
            
            # -----------------------------------------------
            # ---------SOLVE for veff only  ------------------
            # -----------------------------------------------    
            """
            Estimate veff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "veff_only_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            veff = 0.19 # init veff of the generator
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = True#True
            use_forward_grid = True
            use_forward_reff = True
            use_forward_lwc = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
            GT_USE = GT_USE + ' --use_forward_lwc' if use_forward_lwc else GT_USE
            GT_USE = GT_USE + ' --use_forward_reff' if use_forward_reff else GT_USE                
            GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
            GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
            
            # optimization:
            globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
            maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            maxls = 30 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            disp = True # Display optimization progression.
            
            gtol = 1e-16 # Stop criteria for the maximum projected gradient.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            ftol = 1e-16 # Stop criteria for the relative change in loss function.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            loss_type = 'l2'
            # Different loss functions for optimization. Currently only l2 is supported.
            
            #-------------------------------------------------
            # ---- start working with the above parameters:---
            #-------------------------------------------------
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --veff ' + str(veff) +\
                ' --log ' + log_name +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)  
            
              
        if(REC_all):
            
            # -----------------------------------------------
            # ---------SOLVE for lwc only  ------------------
            # -----------------------------------------------    
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. rayleigh scattering (add_rayleigh)
            4. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            log_name = "rec_all_"+log_name_base
            # -----------------------------------------------
            # ---------------- inverse parameters:-----------
            # -----------------------------------------------
            
            stokes_weights = [1.0, 0.0, 0.0, 0.0] # Loss function weights for stokes vector components [I, Q, U, V].
                    
            # Initialization and Medium parameters:
            lwc = 0.01 # init lwc of the generator
            reff = 12 # init reff of the generator
            #veff = 0.15 # init veff of the generator
            
            # init_generator = 'Homogeneous'# it is the CloudGenerator from shdom.generate.py
            init = 'Homogeneous'
            # The mie_base_path is defined at the begining of this script.
            INIT_USE = ' --init '+ init
            add_rayleigh = True
            use_forward_mask = True#True
            use_forward_grid = True
            if_save_gt_and_carver_masks = True
            if_save_final3d = True
            
            GT_USE = ''
            GT_USE = GT_USE + ' --use_forward_veff --lwc_scaling 15 --reff_scaling 0.01'# -----------------                       
            GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
            GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
            GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE               
            GT_USE = GT_USE + ' --save_gt_and_carver_masks' if if_save_gt_and_carver_masks else GT_USE
            GT_USE = GT_USE + ' --save_final3d' if if_save_final3d else GT_USE
            
            # optimization:
            globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
            maxiter = 1000 #1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            maxls = 100 # Maximum number of line search steps (per iteration). For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
            disp = True # Display optimization progression.
            
            gtol = 1e-16 # Stop criteria for the maximum projected gradient.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            ftol = 1e-16 # Stop criteria for the relative change in loss function.
            # For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
            loss_type = 'l2'
            # Different loss functions for optimization. Currently only l2 is supported.
            
            #-------------------------------------------------
            # ---- start working with the above parameters:---
            #-------------------------------------------------
            # ' --veff ' + str(veff) +\
            OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                ' --lwc ' + str(lwc) +\
                ' --reff ' + str(reff) +\
                ' --log ' + log_name +\
                ' --air_path ' + air_path +\
                ' --n_jobs '+ str(n_jobs)+\
                ' --loss_type '+ str(loss_type)+\
                ' --maxls '+ str(maxls)+\
                ' --maxiter '+ str(maxiter)+\
                ' --radiance_threshold '+ str(radiance_threshold)
            
            OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if globalopt else OTHER_PARAMS    
            
            # We have: ground_truth, rte_solver, measurements.
            SCRIPTS_PATH = '../scripts'
            cmd = 'python '+ os.path.join(SCRIPTS_PATH, 'optimize_microphysics_lbfgs.py')+\
                OTHER_PARAMS + GT_USE + INIT_USE
            
            Optimaize1 = subprocess.call( cmd, shell=True)            
              
#-------------------------------------------------
#-------------------------------------------------
#-------------------------------------------------    

    # that the time to show the results in 3D visualization:
    VIS_RESULTS3D = False
    if(VIS_RESULTS3D):    
        """
        The forward_dir id a folder that containes:
        medium, solver, measurements.
        They loaded before. To see the final state, the medium is not 
        enough, the medium_estimator is needed.
        load the measurments to see the rendered images:
        """
        
        
        # what state to load? I prefere the last one!
        logs_dir = os.path.join(forward_dir,'logs')
        logs_prefix = os.path.join(logs_dir,log_name)
        logs_files = glob.glob(logs_prefix + '-*')
        
        times = [i.split('{}-'.format(int(1e3*wavelength_micron)))[-1] for i in logs_files]
        # sort the times to find the last one.
        timestamp = [time.mktime(time.strptime(i,"%d-%b-%Y-%H:%M:%S")) for i in times]
        # time.mktime(t) This is the inverse function of localtime()
        timestamp.sort()
        timestamp = [time.strftime("%d-%b-%Y-%H:%M:%S",time.localtime(i)) for i in timestamp]
        # now, the timestamp are sorted, and I want the last stamp to visualize.
        connector = '{}-'.format(int(1e3*wavelength_micron))
        log2load = logs_prefix+'-'+timestamp[-1]
        # print here the Final results files:
        Final_results_3Dfiles = glob.glob(log2load + '/FINAL_3D_*.mat')
        print("{} files with the results in 3D were created:".format(len(Final_results_3Dfiles)))
        for _file in Final_results_3Dfiles:
            
            print(_file)
            
        # ---------------------------------------------------------
        
        # Don't want to use it now, state2load = os.path.join(log2load,'final_state.ckpt')
        
    
        print("use tensorboard --logdir {} --bind_all".format(log2load))
    
    print("done")
