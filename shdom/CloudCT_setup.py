import os 
import sys
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom 
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 


# -----------------------------------------------------------------
# ------------------------THE CLASSES BELOW------------------------
# -----------------------------------------------------------------
class SpaceMultiView(shdom.MultiViewProjection):
    """
    Inherent from pyshdom MultiViewProjection. We will use the Perspective projection"""
    def __init__(self, imager = None):
        super().__init__()
        assert imager is not None, "You must provied the imager object!" 
        self._sat_positions = None # it will be set by set_satellites() and it is np.array.
        self._lookat_list = None
    
        self._unique_wavelengths_list =  None # if the cameras have different wavelenrgths, this list will have all existing wavelengths. In the bottom line, the rte_solvers depend on that list.
        self._IMAGES_DICT = None
        self._imager = imager
        
        # TODO - to inplement later:
        """
        # what are the existing wavelengths in this setup?
        if(np.isscalar(self._wavelengths_list)):
            self._unique_wavelengths_list = self._wavelengths_list
        else:  
            import itertools
            merged = list(itertools.chain(*self._wavelengths_list))
            unique = list(set(merged))
            self._unique_wavelengths_list = unique
            print("This setup has {} unique wavelengths".format(unique))  
        """      
     
    def update_views(self,names = None):
        """
        This method does the following:
        1. Direct the sateliites to the right look at vectors.
        2. Build the list of Imagers.
        
        """
        # nameas:
        if names is None:
            names = ["sat"+str(i+1) for i in range(len(self._sat_positions))]                 
        
        # we intentialy, work with projections lists and one Imager
        up_list = np.array(len(self._sat_positions)*[0,1,0]).reshape(-1,3) # default up vector per camera.
        for pos,lookat,up,name in zip(self._sat_positions,\
                                  self._lookat_list,up_list,names):
            
            x,y,z = pos
            fov = self._imager.FOV
            nx, ny = self._imager.get_sensor_resolution()
            loop_projection = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
            loop_projection.look_at_transform(lookat, up)
            self.add_projection(loop_projection,name)
            """
            Each projection has:
            ['x', 'y', 'z', 'mu', 'phi'] per pixel and npix, names, resolution.
            """
            
    def set_satellites_position_and_lookat(self,sat_positions,sat_lookats):
        """
        Parameters:
        input:
         - sat_positions: x,y,z of the satellites (each row has x,y,z of some sat).
           np.array shape of (#sats,3)
         - similar for the sat_lookats, it is where the satellites look at.
           np.array shape of (#sats,3)
         
        """
        self._sat_positions = sat_positions
        self._lookat_list = sat_lookats
   
    def find_radiance_thresholds(self):
        """
        User interface to find rigth treshols for the cloud masking (and at the ens Space Carving).
        
        """
        #from matplotlib.widgets import Slider, Button
        #projection_names = self.projection_names
        ## A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
        #USED_CAMERAS = self._measurements.camera
        #RENDERED_IMAGES = self._measurements.images    
        #images_array = np.array(RENDERED_IMAGES)
        ## calculate images maximum:
        #MAXI = images_array.max()
        
        ## currently all the cameras identicle:
        #SATS_NUMBER_SETUP = len(RENDERED_IMAGES)
        pass
        
                
        
        
    def show_measurements(self,radiance_threshold=None,compare_for_test = False):
        """
        show images per all wavelengths.
        use compare_for_test - True if you want to compare the renderings to insure they are nut the same 
        for all wavelengths.
        """
        projection_names = self.projection_names
        # A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
        USED_CAMERAS = self._measurements.camera
        RENDERED_IMAGES = self._measurements.images    
        images_array = np.array(RENDERED_IMAGES)
        # calculate images maximum:
        MAXI = images_array.max()
        
        # currently all the cameras identicle:
        SATS_NUMBER_SETUP = len(RENDERED_IMAGES)
        
        # loop over all existing wavelengths:
        for WI, wavelength in enumerate(self._unique_wavelengths_list):
                
            # PLOT one plot per wavelength:
            # todo for on wavelengths
            fig = plt.figure(figsize=(20, 20))
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(2, int(round(np.ceil(SATS_NUMBER_SETUP/2)))),
                            axes_pad=0.3,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.1
                            )  
            
            for ax, name in zip(grid, projection_names):
                image = self._IMAGES_DICT[name][wavelength].copy()
                if(radiance_threshold is not None):
                    if(isinstance(radiance_threshold, list)):
                        image[image<=radiance_threshold[WI]] = 0
                    else: 
                        image[image<=radiance_threshold] = 0
                    
                ax.set_axis_off()
                im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
                ax.set_title("{}".format(name))
                
            # Since currently the cameras are identicale, nx = self._nx_list[0], the same for ny.
            nx = self._nx_list[0]
            ny = self._ny_list[0]
            title = "$\lambda$={}nm , nx={} , ny={}".format(int(1e3*wavelength),nx,ny)
            cbar = ax.cax.colorbar(im)
            cbar = grid.cbar_axes[0].colorbar(im)
            fig.suptitle(title, size=16,y=0.95)
            
        if(compare_for_test and len(self._unique_wavelengths_list)>1):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            for sat_index, (sat_name,wavelengths_per_sat) in enumerate(zip(self.projection_names,self._wavelengths_list)):
                if(not sat_index == 6):
                    continue
                fig, grid = plt.subplots(1, len(wavelengths_per_sat), figsize=(15, 15))    
                fig.suptitle("Compare to images of {}".format(sat_name), size=16,y=0.85)  
                colors = ['r','g','b','k','m','c']
                colors = colors[:len(wavelengths_per_sat)]
                for color, main_ax , wavelength in zip(colors,grid, wavelengths_per_sat):
                    divider = make_axes_locatable(main_ax)
                    bottom_ax = divider.append_axes("bottom", 2.05, pad=0.7,sharex=main_ax)                    
                    bottom_ax.xaxis.set_tick_params(labelbottom=False)
                    main_ax.axis('off')
                    
                    bottom_ax.set_ylabel('cross section')
                    title = "$\lambda$={}nm".format(int(1e3*wavelength))
                    main_ax.set_title(title)  
                    
                    cur_x = 30#int(0.5*self._nx_list[sat_index])  
                    cur_y = 25#int(0.5*self._ny_list[sat_index])  
                    image = self._IMAGES_DICT[sat_name][wavelength]
                    im = main_ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
                    
                    main_ax.autoscale(enable=True)
                    bottom_ax.autoscale(enable=False)
                    if(wavelength == 1.6):
                        v_line = main_ax.axvline(cur_x, color='r',linestyle = '--')
                        h_line = main_ax.axhline(cur_y, color='g',linestyle = '--')
                        
                        bottom_ax.plot(np.arange(image.shape[0]), image[int(cur_y),:], color='r',linestyle = '--')
                        bottom_ax.set_title('horizontal')
                        
                        bottom_ax.set_title('vertical')                        
                    else:
                        v_line = main_ax.axvline(cur_x, color='r')
                        h_line = main_ax.axhline(cur_y, color='g')
                        
                        bottom_ax.plot(np.arange(image.shape[0]), image[int(cur_y),:], color='r')
                        bottom_ax.set_title('horizontal')
                        
                        bottom_ax.set_title('vertical')
                        
                    bottom_ax.set_ylim(top=MAXI)
                    bottom_ax.autoscale(enable=False)
                    
                cb_ax = fig.add_axes([0.93, 0.41, 0.02, 0.3])
                cbar = fig.colorbar(im, cax=cb_ax)                
            
            
            
    def add_noise(self):
        """
        IMPORTANT:
        Later on, We need to add niose to the measurements
        we can do it here, like: 
        measurements = noise.apply(measurements)
        """              
        pass
        
    def update_measurements(self,sensor=None, projection=None,rte_solver = None, n_jobs=1):
          
        """
        This method renders the images and update the measurements.
        Currently, this method sopports only sensor of type shdom.RadianceSensor()
        Parameters
        ----------
        sensor: shdom.Sensor
            A sensor object
        projection: shdom.Projection
            A projection geometry
        rte_solver: shdom.RteSolver object
            The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).  
            It can be just a rte_solver or if the rendering is for several atmospheres (e.g. multi-spectral rendering),
            It is shdom.RteSolverArray. The camera.render(rte_solver) takes care of the distribution of the rendering to one solver or more.
        n_jobs: int
            how many cores will be used in the backpropogation to gather the radiance from the scene.
        """
        if(not isinstance(rte_solver.wavelength, list)):
            self._unique_wavelengths_list = [rte_solver.wavelength]
        else:
            self._unique_wavelengths_list = rte_solver.wavelength
            
        assert sensor is not None, "You must provied the sensor."
        assert projection is not None, "You must provied the projections."
        assert isinstance(sensor, shdom.sensor.RadianceSensor), "Currently, the measurments supports only sensor of type shdom.sensor.RadianceSensor"
        assert rte_solver is not None, "You must provied the rte_solver/s."
        
        camera = shdom.Camera(sensor, projection)
        # render all:
        images = camera.render(rte_solver, n_jobs=n_jobs)
        # put the images in Measurements object:
        measurements = shdom.Measurements(camera, images=images, wavelength=rte_solver.wavelength)
        self._measurements = measurements
        """
        Here I want to seperate images for different view, wavelength and resolution:
        The variable images has the rendered images for every projection in the list.
        Size of images:
         -The length of images is the number of projections.
         -Each element in images has a matrix with (nx*ny*#wavelengths).
         -The list of all existing wavelengths recorded in self._unique_wavelengths_list.
          All the wavelengths in self._unique_wavelengths_list, have its renderings.
         -Generaly, the lists rte_solver.wavelength and self._unique_wavelengths_list have the same elements.
        """
        IMAGES_DICT = OrderedDict()
        # loop over satellites:
        for sat_index, (image,sat_name,wavelengths_per_sat) in enumerate(zip(images,self.projection_names,self._wavelengths_list)):
            WS_DICT = OrderedDict()
            # loop over wavelengths of the setup, WI index of a wavelength:
            if(isinstance(rte_solver.wavelength, list)):
                
                for WI, wavelength in enumerate(rte_solver.wavelength):
                    if(wavelength in wavelengths_per_sat):
                        WS_DICT[wavelength] = image[:,:,WI]
                    else:
                        WS_DICT[wavelength] = np.zeros(self._nx_list[sat_index],self._ny_list[sat_index]) # zerow image
                
                IMAGES_DICT[sat_name] = WS_DICT
            else:
                WS_DICT = OrderedDict()
                WS_DICT[rte_solver.wavelength] = image
                IMAGES_DICT[sat_name] = WS_DICT
                # if rte_solvers.wavelength =  scalar
                
        self._IMAGES_DICT = IMAGES_DICT
        
  

        
    def show_setup(self,scale = 0.6,axisWidth = 3.0,axisLenght=1.0, FullCone = False):
        """
        Show all the views:
        """
        for view,view_name in zip(self._projection_list,self._names):
            view.show_camera(scale,axisWidth,axisLenght,FullCone) 
            
            t = view.position
            mlab.text3d(t[0]+0.2*scale, t[1], t[2]+0.2*scale, view_name, color=(1,0,0),scale=0.02*scale)        
        
    def Random_choose_N(self,N):
        """
        Choose N satellits randomly.
        Parameters
        ----------
        N - How mauny satellites to choose:int
        
        returns:
        N views: class MultiViewProjection(Projection)
        """
        
        idx = np.random.choice(self._num_projections,N,replace=False)
        # generate N unique numbers in that interval of (self._num_projections)
        new_names = [self._names[idx_i] for idx_i in idx]
        NEW = SpaceMultiView() 
        NEW.set_satellites(self._sat_positions[idx],self._lookat_list[idx])
        NEW.set_commonCameraIntrinsics(self._fov_list[idx[0]],self._nx_list[idx[0]],self._ny_list[idx[0]]) 
        NEW.update_views(new_names)
        
        return NEW
   
    @property
    def projection_names(self):
        return self._names  
    
    
    @property
    def measurements(self):
        return self._measurements     
    
# -----------------------------------------------------------------
# ------------------------THE CLASSES ABOVE----------------------
# -----------------------------------------------------------------    
    
# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS BELOW----------------------
# -----------------------------------------------------------------
def plank(llambda,T=5800):
    h = 6.62607004e-34 # Planck constant
    c = 3.0e8
    k = 1.38064852e-23 # Boltzmann constant
    # https://en.wikipedia.org/wiki/Planck%27s_law
    a = 2.0*h*(c**2)
    b = (h*c)/(llambda*k*T)
    spectral_radiance = a/ ( (llambda**5) * (np.exp(b) - 1.0) )
    return spectral_radiance

    
def StringOfPearls(SATS_NUMBER = 10,orbit_altitude = 500):
    """
    Set orbit parmeters:
         input:
         SATS_NUMBER - int, how many satellite to put?
         
         returns sat_positions: np.array of shape (SATS_NUMBER,3).
         The satellites setup alwas looks like \ \ | / /. 
    """
    r_earth = 6371.0 # km 
    Rsat = orbit_altitude # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R
    Darc = 100# km # distance between adjecent satellites (on arc).
    Dtheta = Darc/R # from the center of the earth.
    
    
    # where to set the sateliites?
    theta_config = np.arange(-0.5*SATS_NUMBER,0.5*SATS_NUMBER)*Dtheta
    theta_config = theta_config[::-1] # put sat1 to be the rigthest
    X_config = r_orbit*np.sin(theta_config)
    Z_config = r_orbit*np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config)
    sat_positions = np.vstack([X_config, Y_config , Z_config]) # path.shape = (3,#sats) in km.

    # find near nadir view:
    # since in this setup y=0:
    near_nadir_view_index = np.argmin(np.abs(X_config))
    near_nadir_view_index
    
    return sat_positions.T, near_nadir_view_index


def Create(SATS_NUMBER = 10,ORBIT_ALTITUDE = 500 ,SAT_LOOKATS=None, Imager_config = None, imager=None, VISSETUP = False):
    
    """
    Create the Multiview setup on orbit direct them with lookat vector and set the Imagers at thier locations + orientations.
    The output here will be a list of Imagers. Each Imager will be updated here with respect to the defined geomtric considerations.
    
    
    Parameters:
    input:
       SATS_NUMBER - the number of satellites in the setup, int.
       ORBIT_ALTITUDE - in km  , float.
       SAT_LOOKATS in km is where each satellite looks at. Type np.array shape of (#sats,3)
       Imager_config - A list of bools with SATS_NUMBER elements. If Imager_config[Index] = True, the imager is in the location of satellite number Index.
       If Imager_config[Index] = False, there is no Imager at the location of satellite number Index.
       The default is None, means all satellites have the Imager.
       
    output:
       MV - SpaceMultiView object. In the bottom line, it is a list of projection with the Imager inside.
    
    """
    sat_positions, near_nadir_view_index = StringOfPearls(SATS_NUMBER = SATS_NUMBER, orbit_altitude = ORBIT_ALTITUDE)
    
    assert imager is not None, "You must provide the Imager object!"
    
    if Imager_config is None:
        Imager_config = SATS_NUMBER*[True]

   
    # --------- start the multiview setup:---------------
    # set lookat list:
    if SAT_LOOKATS is None:
        sat_lookats = np.zeros([len(sat_positions),3])
        
    else:
        
        sat_lookats = SAT_LOOKATS
        
    # Work on the Imagers configaration:
    # the list of Imagers is in the SpaceMultiView class:

    MV = SpaceMultiView(imager) 
    MV.set_satellites_position_and_lookat(sat_positions,sat_lookats) # here we set only the positions and where the satellites are lookin at.   
    MV.update_views()
    
    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000    
    if(VISSETUP):
        MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)    
        mlab.show()
        
        
    return MV, near_nadir_view_index


def old_Create(SATS_NUMBER = 10,ORBIT_ALTITUDE = 500, CAM_FOV = 0.1, CAM_RES = (64,64),SAT_LOOKATS=None, SATS_WAVELENGTHS_LIST = None, SOLAR_FLUX_LIST = None, VISSETUP = False):
    
    """
    Create the Multiview setup on orbit with one camera type.
    Parameters:
    Camera parameters are CAM_FOV, CAM_RES.
    ORBIT_ALTITUDE in km.
    SAT_LOOKATS in km is where each satellite look at. Type np.array shape of (#sats,3)
    SATS_WAVELENGTHS_LIST - Each camera has camera_wavelengths_list in microns.
    SOLAR_FLUX_LIST - Each wavelengths in the SATS_WAVELENGTHS_LIST, is a channel in the camera.
       The images per wavelength should be scaled by the values in SOLAR_FLUX_LIST.
       SOLAR_FLUX_LIST has the same size as the SATS_WAVELENGTHS_LIST.
    """
    sat_positions, near_nadir_view_index = StringOfPearls(SATS_NUMBER = SATS_NUMBER, orbit_altitude = ORBIT_ALTITUDE)
    sat_wavelengths = SATS_WAVELENGTHS_LIST
    solar_flux_list = SOLAR_FLUX_LIST
    # --------- start the multiview setup:---------------
    # set lookat list:
    if SAT_LOOKATS is None:
        sat_lookats = np.zeros([len(sat_positions),3])
        
    else:
        
        sat_lookats = SAT_LOOKATS
        
        
    # set camera's field of view:
    FOV = CAM_FOV # deg
    cnx,cny = CAM_RES

    MV = SpaceMultiView() 
    MV.set_satellites(sat_positions,sat_lookats,sat_wavelengths)
    MV.update_solar_irradiances(solar_flux_list)
    MV.set_commonCameraIntrinsics(FOV,cnx,cny)     
    MV.update_views()
    
    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000    
    if(VISSETUP):
        MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)    
        mlab.show()
        
        
    return MV, near_nadir_view_index

def Prepare_Medium(CloudFieldFile=None, AirFieldFile = None, MieTablesPath=None, 
                   wavelengths_micron=None,wavelength_averaging=False):
    """
    Prepare the medium for the CloudCT simulation:
    1. Generate multi-spectral scatterers for both droplets and air molecules.
    
    inputs:
    CloudFieldFile - file path to the cloud field.
    wavelengths_micron: list or a scalar,
        It is the wavelengths in microns.
    wavelength_averaging: bool,
         If it is False, Mie tables with template of Water_{}nm.scat will be loaded
         If it is True, Mie tables with template of averaged_Water_{}nm.scat will be loaded. It means that
         here we want to use the central wavelength and the Mie tables were calculated by averaging scattering properties over the wavelength band.
    """
    
    # Force/check the wavelengths_micron to be a list:
    if(np.isscalar(wavelengths_micron)):
        wavelengths_micron = [wavelengths_micron]
    else:
        if(not isinstance(wavelengths_micron, list)):
            wavelengths_micron = wavelengths_micron.tolist()
                    
    # Generate multi-spectral scatterers for both droplets and air molecules
    assert CloudFieldFile is not None, "You must provied the cloud field for the simulation."
    droplets = shdom.MicrophysicalScatterer()
    droplets.load_from_csv(CloudFieldFile, veff=0.1)
    
    # Air part
    air = shdom.MultispectralScatterer()
    if(AirFieldFile is None):
        print("You did not provied the air field for the simulation. The atmospher will not include Molecular Rayleigh scattering.")

    # user may tune:
    air_num_points = 20 # Number of altitude grid points for the air volume
    air_max_alt = 5 # in km ,Maximum altitude for the air volume
    # ----------------------------------------------------------------
    # Rayleigh scattering for air molecules
    if(AirFieldFile is not None):
        df = pd.read_csv(AirFieldFile, comment='#', sep=' ')
        altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
        temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
        temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
        air_grid = shdom.Grid(z=np.linspace(0, air_max_alt, air_num_points))
    
    assert MieTablesPath is not None, "You must provied the path of the mie tables for the simulation."    
    for wavelength in wavelengths_micron: 
        if(wavelength_averaging):
            table_path = os.path.join(MieTablesPath,'polydisperse','averaged_Water_{}nm.scat'. \
                                      format(int(1e3 * wavelength)))            
        else:
            table_path = os.path.join(MieTablesPath,'polydisperse','Water_{}nm.scat'. \
                                      format(int(1e3 * wavelength)))
        
        # Molecular Rayleigh scattering
        if(AirFieldFile is not None):
            rayleigh = shdom.Rayleigh(wavelength)
            rayleigh.set_profile(temperature_profile.resample(air_grid))
            air.add_scatterer(rayleigh.get_scatterer())
        
        # Droplet Mie scattering
        mie = shdom.MiePolydisperse()
        mie.read_table(table_path)
        droplets.add_mie(mie)
        print('added mie with wavelength of {}'.format(wavelength))
        
        # here droplets.num_wavelengths = air.num_wavelengths = wavelength_num
        
        
    # Generate an atmospheric medium with both scatterers
    if(AirFieldFile is not None):
        atmospheric_grid = droplets.grid + air.grid
    else:
        atmospheric_grid = droplets.grid
        
    # atmospheric_grid = droplets.grid # in a case I don't ant the air in the simulation.
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    if(AirFieldFile is not None):
        atmosphere.add_scatterer(air, name='air')    
    
    return atmosphere
    


# -----------------------------------------------------------------
# ------------------------THE FUNCTIONS BELOW----------------------
# -----------------------------------------------------------------


def main():
    
    sat_positions, near_nadir_view_index = StringOfPearls()
    # --------- start the multiview setup:---------------
    # set lookat list:
    sat_lookats = np.zeros([len(sat_positions),3])
    # set camera's field of view:
    FOV = 0.1 # deg
    cnx,cny = (64,64)
    
    # assume band:
    setup_wavelengths_list = len(sat_positions)*[[0.66]]
    
    
    MV = SpaceMultiView() 
    MV.set_satellites(sat_positions,sat_lookats,setup_wavelengths_list)
    MV.set_commonCameraIntrinsics(FOV,cnx,cny) 
    MV.update_views()
    
    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000    
    VISSETUP = True
    if(VISSETUP):
        MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        figh = mlab.gcf()
        mlab.orientation_axes(figure=figh)
        
    if(VISSETUP):    
        mlab.show()
        
        
        
if __name__ == '__main__':
    main()