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
    start_reff = 0.05 # Starting effective radius [Micron]
    end_reff = 25.0
    num_reff = 100
    start_veff = 0.01
    end_veff = 0.2
    num_veff = 50
    radius_cutoff = 65.0 # The cutoff radius for the pdf averaging [Micron]
    
    
    # --------------------------------------------------------------
    wavelength_list = [wavelength_micron]# I still work in monochromatic, TODO move it to poly.
    assert wavelength_micron is not None, "You must provied the wavelength"
    
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
                
        wavelength_list = wavelength_list_final
            
    
    if(not (wavelength_list==[])):
        
        # importing functools for reduce() 
        import functools  
        # importing operator for operator functions 
        import operator 
        
        wavelength_string = functools.reduce(operator.add,[str(j)+" " for j in wavelength_list]).rstrip()
        wavelength_arg = ' --wavelength '+wavelength_string
        
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
                wavelength_arg
        
        Generate_Mie_scat = subprocess.call( cmd, shell=True)
        
    print('Mie table is calculated.')

    print(20*'-')
    print(20*'-')
    mie_base_path = where_to_check_path+'/Water_{}nm.scat'.\
        format(int(1e3*wavelength_micron))
    
    return mie_base_path
    
# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS ABOVE----------------------
# -----------------------------------------------------------------

# visualization params:
VISSETUP = True
scale = 500
axisWidth = 0.02
axisLenght = 5000  

# orbit altitude:
Rsat = 500 # km
wavelength_micron = 0.672
sun_azimuth = 45
sun_zenith = 150
"""
Check if mie tables exist, if not creat them, if yes skip it is long process.
table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
"""
where_to_check_path = './mie_tables/polydisperse'
mie_base_path = CALC_MIE_TABLES(where_to_check_path,\
                wavelength_micron)
# for example, mie_base_path = './mie_tables/polydisperse/Water_672nm.scat'



# forward_dir, where to save evrerything that is related to forward model:
forward_dir = './experiments/LES_cloud_field_rico_Water_{}nm/monochromatic'.format(int(1e3*wavelength_micron))
# invers_dir, where to save evrerything that is related to invers model:
invers_dir = forward_dir
log_name = 'easiest_rico32x37x26_{}'.format(int(1e3*wavelength_micron))
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
air_grid = shdom.Grid(z=np.linspace(0, air_num_points, air_max_alt))
rayleigh = shdom.Rayleigh(wavelength=wavelength_micron)
rayleigh.set_profile(temperature_profile.resample(air_grid))
air = rayleigh.get_scatterer()

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
SATS_NUMBER = 10
VISSETUP = False
# not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
# tuning is applied by the variavle LOOKAT.
LOOKAT = CENTER_OF_MEDIUM_BOTTOM
LOOKAT[2] = 0.5*nx*dz # tuning

SAT_LOOKATS = np.array(SATS_NUMBER*LOOKAT).reshape(-1,3)       

# create CloudCT setup:
CloudCT_VIEWS = CloudCT_setup.Create(\
    SATS_NUMBER = SATS_NUMBER,ORBIT_ALTITUDE = Rsat, \
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
    numerical_params = shdom.NumericalParameters(num_mu_bins=16,num_phi_bins=32,
                                                 split_accuracy=0.000001,max_total_mb=100000000.0,solution_accuracy=0.00001)
    # Calculate irradiance of the spesific wavelength:
    # use plank function:
    temp = 5900 #K
    L_TOA = 6.8e-5*1e-9*plank(1e-6*wavelength_micron,temp) # units fo W/(m^2)
    Cosain = np.cos(np.deg2rad((180-sun_zenith)))
    solar_flux = L_TOA*Cosain
    
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
    images = camera.render(rte_solver, n_jobs=40)
    
    projection_names = CloudCT_VIEWS.projection_names
    # calculate images maximum:
    images_array = np.array(images)
    MAXI = images_array.max()
    
    # nice plot alternative:
    fig = plt.figure(figsize=(20, 20))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, int(np.ceil(SATS_NUMBER/2))),
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
    SEE_IMAGES = False
    
    MICROPHYSICS = False
    
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
        f, axarr = plt.subplots(2, int(np.ceil(SATS_NUMBER/2)), figsize=(20, 20))
        axarr = axarr.ravel()
        projection_names = CloudCT_VIEWS.projection_names
        # calculate images maximum:
        images_array = np.array(RENDERED_IMAGES)
        MAXI = images_array.max()
        
        index = 0
        for ax, image,name in zip(axarr, RENDERED_IMAGES, projection_names):
            
            
            im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
            #image_gamma = (image/np.max(image))**0.5
            #ax.imshow(image_gamma,cmap='gray')
            if(index > 0):
                ax.axis('off')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)        
            f.colorbar(im, cax=cax, orientation='vertical')
                         
            ax.set_title("{}".format(name))
            index = index + 1
            
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
        3. cloud mask (use_forward_mask)
        4. known albedo (use_forward_albedo)
        5. rayleigh scattering (add_rayleigh)
        
        """
        # -----------------------------------------------
        # ---------------- inverse parameters:-----------
        # -----------------------------------------------
        n_jobs = 72
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
        GT_USE = ''
        GT_USE = GT_USE + ' --add_rayleigh' if add_rayleigh else GT_USE
        GT_USE = GT_USE + ' --use_forward_mask' if use_forward_mask else GT_USE
        GT_USE = GT_USE + ' --use_forward_grid' if use_forward_grid else GT_USE
        GT_USE = GT_USE + ' --use_forward_albedo' if use_forward_albedo else GT_USE
        GT_USE = GT_USE + ' --use_forward_phase' if use_forward_phase else GT_USE
        
        # (use_forward_mask, use_forward_grid, use_forward_albedo, use_forward_phase):
        # Use the ground-truth things. This is an inverse crime which is 
        # usefull for debugging/development.    
        radiance_threshold = 0.05 # Threshold for the radiance to create a cloud mask.
        # Threshold is either a scalar or a list of length of measurements.
        
        # The log_name defined above
        # Write intermediate TensorBoardX results into log_name.
        # The provided string is added as a comment to the specific run.
        
        # note that currently, the invers_dir = forward_dir.
        # In forward_dir, the forward modeling parameters are be saved.
        # If invers_dir = forward_dir:
        # The invers_dir directory will be used to save the optimization results and progress.
        
        # optimization:
        globalopt = False # Global optimization with basin-hopping. For more info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html.
        maxiter = 1000 # Maximum number of L-BFGS iterations. For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.
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
    
    
    else:
        pass
        
        
        
        
    #import sys
    #sys.path.append("../scripts")    
    #import optimize_extinction_lbfgs as OptimizExt
    #import argparse
    
    #OptimizationScript = OptimizExt.OptimizationScript(scatterer_name='cloud')
    #OptimizationScript.parse_arguments()
    #OptimizationScript.main()
    #parser = argparse.ArgumentParser()
    #args = parser.parse_args()
    #parser = OptimizationScript.optimization_args(parser)
    #parser = OptimizationScript.medium_args(parser)
    # Initialize air and cloud medium:
    #CloudGenerator = getattr(shdom.generate, init)
    #cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
    
    #args.air_path = air_path
    #args.air_max_alt = air_max_alt
    #args.air_num_points = air_num_points
    #AirGenerator = shdom.generate.AFGLSummerMidLatAir
    #air_generator = AirGenerator(air_path,air_max_alt,air_num_points)
    
    print("use tensorboard --logdir {}/logs/{} --bind_all".format(invers_dir,log_name))
    
    print("done")