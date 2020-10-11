import os 
import sys
# import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom 
from shdom import float_round, core
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
import time
import glob
# importing functools for reduce() 
import functools  
# importing operator for operator functions 
import operator 
import pickle
import warnings

# -----------------------------------------------------------------
# ------------------------THE CLASSES BELOW------------------------
# -----------------------------------------------------------------
class SpaceMultiView_Measurements(object):
    """
    This class differes from the Measurements defined in sensor.py.
    The Measurments here describs "real" measurments as were taken be an Images from space.
    
    In addition this class take care of the renderings with different (and mess) channels and different Imagers.
    """
    def __init__(self, setup_of_views_list = None):
        """
        Parameters:
        input:
         - setup_of_views_list: TODO 
         
         """
        assert setup_of_views_list is not None, "You must provied the list of setup views."
        # delete views setup if the imager is inactive in al the views:
        self._setup_of_views_list = []
        for setup_of_views in setup_of_views_list:
            if(setup_of_views.num_projections > 0):
                
                self._setup_of_views_list.append(setup_of_views)
                
        self._imagers_unique_wavelengths_list = [float_round(w.imager.centeral_wavelength_in_microns) for w in self._setup_of_views_list] # Actualy the list may be not uniqe if for instance 2 imagers have the same centeral wavelength. 
        # wavelengths in microns.
        self._Images_per_imager = None
        self._Radiances_per_imager = None
        self._radiance_to_graylevel_scales = len(setup_of_views_list)*[1]
        
        self._sensor_type = None # Currently, it sopports only sensor of type shdom.RadianceSensor().
        self._shdom_sensor = None
        # TODO - consider stocks sensor in the future.
        
    def resample_rays_per_pixel(self):
        """
        As we want random sampling of rays in a pixel, we should bettet resample the rays between different rednerings.
        This method does not change the number of ray that will be sampled per pixel.
        """
        for projection_list_per_imager in self._setup_of_views_list:
            projection_list_per_imager.resample_rays_per_pixel()
            
        
    def simulate_measurements(self,rte_solvers = None,n_jobs = 1, IF_APPLY_NOISE = False, IF_SCALE_IDEALLY=False, IF_REDUCE_EXPOSURE=False ):
        """
        This method renders the images and update the measurements.
        It aslo gets the rte_solvers object. The rte_solvers can be just a rte_solver if the imagers have the same central wavelenght.
        In that case, one rte_solver that used one wavelength should be enough.   
        
        Currently, this method sopports only sensor of type shdom.RadianceSensor().
        TODO - add stocks sensor in the future.
        Parameters
        ----------
        Input:
        rte_solver: shdom.RteSolver object
            The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).  
            It can be just a rte_solver or if the rendering is for several atmospheres (e.g. multi-spectral rendering),
            It is shdom.RteSolverArray. The camera.render(rte_solver) takes care of the distribution of the rendering to one solver or more.

        n_jobs: int
            How many cores will be used in the backpropogation to gather the radiance from the scene.
        
        IF_APPLY_NOISE - bool, if it is True, apply noise.
        
        IF_SCALE_IDEALLY - bool, if it is True, ignore imager parameters that plays in the convertion of
        radiance to electrons. It may be used just for simulations of ideal senarios or debug.
        
        IF_REDUCE_EXPOSURE - bool, if it is True, it redusec the exposure time or does not change the exposure time.
        It reduces the exposure time the current exposure time makes saturation e.g. reaches full well.
        It does not change the exposure time if the current exposure time does not makes saturation.
        
        
        """ 
        if isinstance(rte_solvers, shdom.RteSolverArray):
            solvers_unique_wavelengths_list = rte_solvers.wavelength
            if np.isscalar(solvers_unique_wavelengths_list):
                solvers_unique_wavelengths_list = [solvers_unique_wavelengths_list]            
        else:
            rte_solvers = [rte_solvers]
            solvers_unique_wavelengths_list = [rte_solvers[0].wavelength]

                    
        if(IF_SCALE_IDEALLY):# protection from user confussion.
            IF_REDUCE_EXPOSURE = False
            
        sensor=shdom.RadianceSensor()# Currently, this method sopports only sensor of type shdom.RadianceSensor().
        self._sensor_type = sensor.type
        
        # Separate solvers by wavelenghts:
        assert rte_solvers is not None, "You must provied the rte_solver/s."
        Solvers_dict = dict(zip(solvers_unique_wavelengths_list, rte_solvers))
        self._Images_per_imager = OrderedDict()
        self._Radiances_per_imager = OrderedDict()
        
        # Meanwhile use regular pyshdoms rendering to minimaize mess in the rendering refactoring.
        for wavelength in Solvers_dict.keys():
            # Each setup views instance has its own imager.
            # find and connect the imager to the rte solver be matching the wavelengths. 
            # per imager is equivalent to per wavelengths since every imager has its central wavelength
            if(wavelength in self._imagers_unique_wavelengths_list):
                imager_index = self._imagers_unique_wavelengths_list.index(wavelength)
                projections = self._setup_of_views_list[imager_index]
                assert projections.num_projections > 0 , "Inactive imager appears in the list of views."
                
                rte_solver = Solvers_dict[wavelength]
            else:
                warnings.warn("It is very strange that the solver does not have a wavelength that exists in the images list.\nMaybe you deactivate an imager? if not, you have a bug in the setup definition.")
                # raise Exception("It is very strange that the solver does not have a wavelength that exists in the images list.")
            
            print('The rendering is beeing done for centeral_wavelength of {}nm.\n'.format(wavelength))
            camera = shdom.Camera(sensor, projections)
            # render all view per wavelength:
            self._Radiances_per_imager[imager_index] = camera.render(rte_solver, n_jobs=n_jobs)   
            # TODO - Scale the images such that it will be in grayscales   
            # Here the regular pyshdoms rendering is finished, now we scale the images with respect to the imager properties:
            # The images are images of radiaces. 
            # Convert radiance to electrons to grayscale:
            images_list_per_imager = self._Radiances_per_imager[imager_index].copy()
            if(not isinstance(images_list_per_imager, list)):
                images_list_per_imager = [images_list_per_imager]
              
            if(IF_REDUCE_EXPOSURE):
                projections.imager.adjust_exposure_time(images_list_per_imager, C = 0.9)# C = 0.9 - the imager will reach 90 procent of the full well.
                 
            self._Images_per_imager[imager_index], radiance_to_graylevel_scale = projections.imager.convert_radiance_to_graylevel(images_list_per_imager,IF_APPLY_NOISE=IF_APPLY_NOISE,IF_SCALE_IDEALLY=IF_SCALE_IDEALLY)
            # the lines below does the following: It scale back the grayscale values to radiances but it does that after nosie addition.
            # TODO - Ensure the feasability of that step with Yoav.
            images_in_grayscale = self._Images_per_imager[imager_index]
                           
                    
            self._Images_per_imager[imager_index] = [(i/radiance_to_graylevel_scale) for i in images_in_grayscale] # 
                         
                    
            self._radiance_to_graylevel_scales[imager_index] = radiance_to_graylevel_scale
        
    
    def show_measurments(self, radiance_threshold_dict = None,title_content=''):
        """
        TODO - help
        TODO - use radiance_threshold_dict which is a dictionary: keys air the imager indexes, values are intensity tresholds per view.
        You can use radiance_threshold_dict in the visualization just to see how the images (in grayscale) look like with that tresholds.
        """
        if(radiance_threshold_dict is None):
            radiance_threshold_dict = dict()
            for imager_index in self._Images_per_imager.keys():
                projections = self._setup_of_views_list[imager_index]
                radiance_threshold_dict[imager_index] = projections.num_projections*[0]
        
        for imager_index in self._Images_per_imager.keys():
            projections = self._setup_of_views_list[imager_index]
            radiance_thresholds = radiance_threshold_dict[imager_index] # tresholds per imager
            
            ncols = 5 # colums, in subplots
            nrows_ncols = [int(np.ceil(projections.num_projections/ncols)), int(ncols)]
            
            if(self._Images_per_imager[imager_index] is not None):
                
                images = self._Images_per_imager[imager_index].copy()
                # calculate images maximum:
                images_array = np.array(images)
                #MAXI = images_array.max()
                MAXI = projections.imager.get_gray_lavel_maximum()
                
                if(nrows_ncols[0] == 1):
                    nrows_ncols[1] = min(nrows_ncols[1],projections.num_projections)
                
                if(nrows_ncols == [1,1]):
                    fig = plt.figure(figsize=(8, 8))
                    ax = plt.gca()
                    
                    image = images[0].copy()*\
                        self._radiance_to_graylevel_scales[imager_index] # to set the images in the grayscale level.
                    image[image<=radiance_thresholds[0]] = 0
                    assert len(images) == 1, "Imposible that there is more than 1 image here."
                    im = plt.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
                    
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)                
                else:   
                    fig = plt.figure(figsize=(20, 20))
                    grid = AxesGrid(fig, 111,
                                    nrows_ncols=nrows_ncols,
                                    axes_pad=0.3,
                                    cbar_mode='single',
                                    cbar_location='right',
                                    cbar_pad=0.1
                                    )  
                    
                    # show all
                    for projection_index, (ax, name) in enumerate(zip(grid, projections.projection_names)):
                        image = images[projection_index].copy()
                        image[image<=radiance_thresholds[projection_index]] = 0
                        image *= self._radiance_to_graylevel_scales[imager_index] # to set the images in the grayscale level..
                        
                        
                        ax.set_axis_off()
                        im = ax.imshow(image,cmap='gray',vmin=0, vmax=MAXI)
                        ax.set_title("{}".format(name))
                    
                    # super title:    
                    
                    # Since currently the view per same imager have identicale nx and ny:
                    nx, ny =  image.shape
                    title = f"Sun Zenith {title_content}, Imager type is {projections.imager.short_description()}, nx={nx} , ny={ny}"
                    cbar = ax.cax.colorbar(im)
                    cbar = grid.cbar_axes[0].colorbar(im)
                    fig.suptitle(title, size=16,y=0.95)
                    #plt.savefig(f'sun_zenith_cloud_retrievals/cloud_retrievals_sun_zenith_{title_content}.png')
                    #plt.close()

                
    def convert2regular_measurements(self):
        """
        Converts the CloudCT measurements to regular measurements.
        Thats to say:
        SpaceMultiView_Measurements -> Measurements class in shdom package.
        """
        #projection = shdom.MultiViewProjection()
        sensor=shdom.RadianceSensor()# Currently, this method sopports only sensor of type shdom.RadianceSensor().
        assert len(self._imagers_unique_wavelengths_list) == 1, "If the case is polychromatic, it must be a mistake, you should not try to convert the measurments. It works only for monochromatic!"
        projections = self._setup_of_views_list[0]

        camera = shdom.Camera(sensor, projections)
        images = self._Radiances_per_imager[0]
        wavelength = self._imagers_unique_wavelengths_list[0]
        # put the images in Measurements object:
        measurements = shdom.Measurements(camera, images=images, wavelength=wavelength)
        return measurements
    
    
    def save(self, path):
        """
        Save SpaceMultiView_Measurements to file.
    
        Parameters
        ----------
        path: str,
            Full path to file. 
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self, -1))
        file.close()
          
    def load(self, path):
        """
        Load SpaceMultiView_Measurements from file.

        Parameters
        ----------
        path: str,
            Full path to file. 
        """        
        file = open(path, 'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)
        
      
      
    @property   
    def setup(self):
        return self._setup_of_views_list
    
    
    @property   
    def images(self):
        return self._Images_per_imager 
    
    @property
    def sensor_type(self):
        return self._sensor_type    
 
    @property
    def shdom_sensor(self):
        return self._shdom_sensor
    
    @property
    def num_channels(self):
        """
        It is the number of wavelengths or bands used in the setup.
        In some way it is the number of channels used in the whole setup. In addition, to be similar to original pyshdom, we call this as num_channels.
        """
        return len(self._imagers_unique_wavelengths_list)
    
    def get_channels_of_imagers(self):
        """
        TODO
        """
        return self._imagers_unique_wavelengths_list
# -------------------------------------------------------------------------------------------------------------------

class SpaceMultiView(shdom.MultiViewProjection):
    """
    Inherent from pyshdom MultiViewProjection. We will use the Perspective projection.
    It encapsulates the geometry of the setup and the imager which is used in the same setup. Only one imager can be used here. Thus SpaceMultiView object, should be defined per imager.
    """
    def __init__(self, imager = None,Imager_config=None, samples_per_pixel = 1, rigid_sampling = False):
        super().__init__()
        assert imager is not None, "You must provied the imager object!" 
        self._sat_positions = None # it will be set by set_satellites() and it is np.array.
        self._lookat_list = None
    
        self._unique_wavelengths_list =  None # if the cameras have different wavelenrgths, this list will have all existing wavelengths. In the bottom line, the rte_solvers depend on that list.
        self._IMAGES_DICT = None
        self._imager = imager
        self._num_channels = 1 # number of channels per the self._imager 
        # TODO- consider to use num_channels>1 per one imager.
        self._Imager_config = Imager_config    
        self._rays_per_pixel = samples_per_pixel
        self._rigid_sampling = rigid_sampling
        
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
    def resample_rays_per_pixel(self):
        """
        As we want random sampling of rays in a pixel, we should bettet resample the rays between different rednerings.
        This method does it.
        """
        assert self.num_projections >0 , "You can not resamples rays per pixels since you have not defined the projections yet."
        for view_index, (view,view_name) in enumerate(zip(self._projection_list,self._names)):
            if((view.samples_per_pixel  > 1) and not self._rigid_sampling):
                
                #print('View name {} resample its ray directions.'.format(view_name))
                #will not work: self._projection_list[view_index].resample_rays_per_pixel()
                #super().__init__()
                self._num_projections = 0
                self._projection_list = []
                self._names = []                
                self.update_views()
        
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
        for pos,lookat,up,samples_per_pixel,name,Flag in zip(self._sat_positions,\
                                  self._lookat_list,up_list,self._rays_per_pixel,names,self._Imager_config):
            
            # Flag is true if the Imager in an index is defined.
            # Only in that case its projection will be in the set.
            if(Flag):
                
                x,y,z = pos
                fov = np.rad2deg(self._imager.FOV) 
                nx, ny = self._imager.get_sensor_resolution()
                loop_projection = shdom.PerspectiveProjection(fov, nx, ny, x, y, z,samples_per_pixel,self._rigid_sampling)
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
        if(self._rays_per_pixel == 1):
            self._rays_per_pixel = np.tile(self._rays_per_pixel,len(self._lookat_list))
        
        else:
            self._rays_per_pixel = np.tile(self._rays_per_pixel,len(self._lookat_list))
            """
            I don't use different samples_per_pixel values among the views.
            It is not adapted in the inverse part.
            But It is well adapted in the forward part.
            TODO - adapt it in the inverse part.
            """
            # scale the samples_per_pixel with the distance of the satellite from lookat:
            #distances = [np.linalg.norm(i-j) for (i,j) in zip(self._sat_positions,self._lookat_list)]
            #close_distance = min(distances)
            #scaling = distances/close_distance
            #self._rays_per_pixel = np.array([int(i*j) for (i,j) in zip(scaling,self._rays_per_pixel)])


        if self._Imager_config is None:
            print('The Imager of {} will be createds for every satellite.'.format(self._imager.short_description()))
            self._Imager_config = len(self._sat_positions)*[True]        
   
    def find_radiance_thresholds(self):
        """
        User interface to find rigth treshols for the cloud masking (and at the ens Space Carving).
        
        """
        
        pass
        
            
    def add_orientation_noise(self):
        """
        IMPORTANT:
        Later on, We need to add noise on the pointing of the satellites.
        """              
        pass
        

    def show_setup(self,scale = 0.6,axisWidth = 3.0,axisLenght=1.0, FullCone = False):
        """
        Show all the views:
        """
        for view,view_name in zip(self._projection_list,self._names):
            view.show_camera(scale,axisWidth,axisLenght,FullCone) 
            
            t = view.position
            # mlab.text3d(t[0]+0.2*scale, t[1], t[2]+0.2*scale, view_name, color=(1,0,0),scale=0.02*scale)
        
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
    def imager(self):
        return self._imager
    
    @property
    def num_channels(self):
        return self._num_channels
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


def Create(SATS_NUMBER = 10,ORBIT_ALTITUDE = 500 ,SAT_LOOKATS=None, Imager_config = None, imager=None, samples_per_pixel = 1, rigid_sampling = False, VISSETUP = False):
    
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
       samples_per_pixel - How much rays to simulate per one pixel.
         - TODO - use samples_per_pixel per view point e.g. closer views have less samples_per_pixel than farther views
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

    MV = SpaceMultiView(imager,Imager_config,samples_per_pixel,rigid_sampling) # it is the geomerty for spesific imager
    MV.set_satellites_position_and_lookat(sat_positions,sat_lookats) # here we set only the positions and where the satellites are lookin at.   
    MV.update_views()
    
    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000    
    if(VISSETUP):
        MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        # figh = mlab.gcf()
        # mlab.orientation_axes(figure=figh)
        # mlab.show()
        
        
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
    #MV.resample_rays_per_pixel()
    
    # visualization params:
    scale = 500
    axisWidth = 0.02
    axisLenght = 5000    
    if(VISSETUP):
        MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        # figh = # mlab.gcf()
        # mlab.orientation_axes(figure=figh)
        # mlab.show()
        
        
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
def old_connect_to_rte_solvers(self,rte_solvers):
    """
    This method conect the class to rte_solvers object.
    The rte_solvers can be just a rte_solver if the imagers have the same central wavelenght.
    In that case, one rte_solver that used one wavelength should be enough.   
    Parameters:
    Input:
    rte_solver: shdom.RteSolver object
        The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).  
        It can be just a rte_solver or if the rendering is for several atmospheres (e.g. multi-spectral rendering),
        It is shdom.RteSolverArray. The camera.render(rte_solver) takes care of the distribution of the rendering to one solver or more.
    
    """
    if isinstance(rte_solvers, shdom.RteSolverArray):
        self._rte_solvers = rte_solvers
        self._solvers_unique_wavelengths_list = self._rte_solvers.wavelength
    else:
        self._rte_solvers = [rte_solvers]
        self._solvers_unique_wavelengths_list = [self._rte_solvers[0].wavelength]
        

def main():
    
    sat_positions, near_nadir_view_index = StringOfPearls(SATS_NUMBER = 10, orbit_altitude = 500)
    # --------- start the multiview setup:---------------
    i_index = 9
    
    X_path = sat_positions[:,0]
    Z_path = sat_positions[:,2]
    offNadirAngles = np.rad2deg(np.arctan(X_path/Z_path)) # from the ground.
    print(offNadirAngles)
    #print(X_path[i_index])
    #print(X_path)
    #print(Z_path)
    #DPHI = np.deg2rad(3.3)
    
    
    ## visualization params:
    #scale = 500
    #axisWidth = 0.02
    #axisLenght = 5000    
    #VISSETUP = True
    #if(VISSETUP):
        #MV.show_setup(scale=scale ,axisWidth=axisWidth ,axisLenght=axisLenght,FullCone = True)
        #figh = mlab.gcf()
        #mlab.orientation_axes(figure=figh)
        
    #if(VISSETUP):    
        #mlab.show()
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
        # figh = mlab.gcf()
        # mlab.orientation_axes(figure=figh)
        
    # if(VISSETUP):
    #     mlab.show()
        
        
        
if __name__ == '__main__':
    main()