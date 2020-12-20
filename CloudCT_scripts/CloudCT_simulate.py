import csv
import gc
import logging
import matplotlib.pyplot as plt
import yaml
import sys
from itertools import chain
import scipy.io as sio
import copy

from shdom import CloudCT_setup, plank
from shdom.CloudCT_Utils import *
import random

def main(CloudFieldFile = None, Init_dict = None, Prefix = None, init = None, mat_path = None, logger=None):

    if not logger:
        logger = create_and_configer_logger(log_name='run_tracker.log')
    logger.debug("--------------- New Simulation ---------------")

    #run_params = load_run_params(params_path="run_params_rico.yaml")
    run_params = load_run_params(params_path="run_params.yaml")

    if CloudFieldFile is not None:
        run_params['CloudFieldFile'] = CloudFieldFile
        
    if Init_dict is not None:
        run_params['inverse_options']['lwc'] = Init_dict['lwc']
        run_params['inverse_options']['reff'] = Init_dict['reff']
        
    if Prefix is not None:
        run_params['Log_Prefix'] = Prefix   
    
    if init is not None:
        run_params['inverse_options']['init'] = init
        if init == 'FromMatFile':
            run_params['inverse_options']['mat_path'] = mat_path
        else:
            raise Exception('Not implemented yet.')


    #run_params['inverse_options']['reff_smoothness_const'] = reff_smoothness_const 
    # run_params['sun_zenith'] = sun_zenith # if you need to set the angle from main's input
    # logger.debug(f"New Run with sun zenith {run_params['sun_zenith']} (overrides yaml)")
    

    """
        Here load the imagers, the imagers dictates the spectral bands of the rte solver and rendering.
        Since the spectrum in this script referes to narrow bands,
        the nerrow bands will be treated in the following maner:
        We will use the wavelength averaging of shdom. It averages scattering properties over the wavelength band.
        Be careful, the wavelengths in Imager methods are in nm. Pyshdom the wavelength are usualy in microns.

        THe imagers aslo dictate the ground spatial resolution (GSD).
    """
    MieTablesPath = os.path.abspath("../mie_tables")

    # Setup the imagers, can be simple!
    vis_imager, vis_wavelength_range, vis_pixel_footprint, vis_imager_config = setup_imager(
        imager_options=run_params['vis_options'],
        run_params=run_params,
        MieTablesPath=MieTablesPath,
        simple_type='vis')

    USESWIR = True if run_params['swir_options']['true_indices'] else False
    # At least one VIS imager will be on.
    # SWIR imager can be canlceled at all. If you do true_indices: [], no swir imager will be simulated (in the run_params file).

    if USESWIR:
        swir_imager, swir_wavelength_range, swir_pixel_footprint, swir_imager_config = setup_imager(
            imager_options=run_params['swir_options'],
            run_params=run_params,
            MieTablesPath=MieTablesPath,
            simple_type='swir')

    SATS_NUMBER_SETUP = run_params['SATS_NUMBER_SETUP']
    
    cloud_name = run_params['CloudFieldFile'].split('/')[-1]
    # where to save the forward outputs:
    if USESWIR:
        forward_dir = f'../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_{int(1e3 * vis_wavelength_range[0])}-' \
                      f'{int(1e3 * vis_wavelength_range[1])}nm_active_sats_{SATS_NUMBER_SETUP}_' \
                      f'GSD_{int(1e3 * vis_pixel_footprint)}m_and' \
                      f'_SWIR_{int(1e3 * swir_wavelength_range[0])}-{int(1e3 * swir_wavelength_range[1])}' \
                      f'nm_active_sats_{SATS_NUMBER_SETUP}_GSD_{int(1e3 * swir_pixel_footprint)}m' \
                      f"_LES_cloud_field_{cloud_name.split('_')[0]}"
    else:
        forward_dir = f'../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_{int(1e3 * vis_wavelength_range[0])}-' \
                      f'{int(1e3 * vis_wavelength_range[1])}nm_active_sats_{SATS_NUMBER_SETUP}_' \
                      f'GSD_{int(1e3 * vis_pixel_footprint)}m' \
                      f"_LES_cloud_field_{cloud_name.split('_')[0]}"

    if run_params['use_cal_uncertainty']:
        forward_dir = forward_dir + '_cal_uncertainty_{}'.format(random.randint(1,1000))
        if run_params['uncertainty_options']['use_bias']:
            forward_dir = forward_dir + '_use_bias_{}'.format(run_params['uncertainty_options']['max_bias'])
        if run_params['uncertainty_options']['use_gain']:
            forward_dir = forward_dir + '_use_gain_{}'.format(run_params['uncertainty_options']['max_gain'])
                
    
    
        
    # inverse_dir, where to save everything that is related to invers model:
    inverse_dir = forward_dir  # TODO not in use
    #log_name_base = f"active_sats_{SATS_NUMBER_SETUP}_{cloud_name}"
    lwc_scaling_val = run_params['inverse_options']['lwc_scaling_val']
    reff_scaling_val = run_params['inverse_options']['reff_scaling_val']
    log_name_base = f"prec_lwc_{lwc_scaling_val}_prec_reff_{reff_scaling_val}_{cloud_name}"
    log_name_base = 'force_1d_reff_' + log_name_base if run_params['inverse_options']['force_1d_reff'] else log_name_base    
    
    log_name_base = 'reff_smoothness_with_' + str(run_params['inverse_options']['reff_smoothness_const']) + log_name_base
    log_name_base = log_name_base + '_precond_grad_' if run_params['inverse_options']['precond_grad'] else log_name_base
    
    if (not run_params['inverse_options']['use_forward_mask']):
        log_name_base = 'space_curv_' + log_name_base
    # Write intermediate TensorBoardX results into log_name.
    # The provided string is added as a comment to the specific run.

    vizual_options = run_params['vizual_options']  # visualization params
    # -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
    # Path to csv file which contains temperature measurements or None if the atmosphere will not consider any air.
    AirFieldFile = run_params['AirFieldFile'] if not run_params['forward_options']['CENCEL_AIR'] else None

    # If we do not use simple imager we probably use imager with a band so:
    wavelength_averaging = False if run_params['USE_SIMPLE_IMAGER'] else True

    # central wavelengths
    if USESWIR:
        wavelengths_micron = [vis_imager.centeral_wavelength_in_microns, swir_imager.centeral_wavelength_in_microns]
        # wavelengths_micron will hold the vis swir wavelengths. It will be convinient to use in some loops
    else:
        wavelengths_micron = [vis_imager.centeral_wavelength_in_microns]

    atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile=run_params['CloudFieldFile'],
                                              AirFieldFile=AirFieldFile,
                                              air_num_points = run_params['forward_options']['air_num_points'],
                                              air_max_alt = run_params['forward_options']['air_max_alt'],
                                              MieTablesPath=MieTablesPath,
                                              wavelengths_micron=wavelengths_micron,
                                              wavelength_averaging=wavelength_averaging)

    droplets = copy.deepcopy(atmosphere.get_scatterer('cloud'))
    

    # -----------------------------------------------
    # ---------Set relevant camera parameters. ------
    # ---For that we need some mediume sizes --------
    # -----------------------------------------------
    droplets_grid = droplets.grid
    dx, dy = droplets_grid.dx, droplets_grid.dy

    nx, ny, nz = droplets_grid.nx, droplets_grid.ny, droplets_grid.nz

    Lx = droplets_grid.bounding_box.xmax - droplets_grid.bounding_box.xmin
    Ly = droplets_grid.bounding_box.ymax - droplets_grid.bounding_box.ymin
    Lz = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
    L = max(Lx, Ly)

    Lz_droplets = droplets_grid.bounding_box.zmax - droplets_grid.bounding_box.zmin
    dz = Lz_droplets / (nz - 1)

    # USED FOV, RESOLUTION and SAT_LOOKATS:
    # cny x cnx is the camera resolution in pixels
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['Rsat'])))

    vis_cnx = vis_cny = int(np.floor(L / vis_pixel_footprint))
    if USESWIR:
        swir_cnx = swir_cny = int(np.floor(L / swir_pixel_footprint))

    CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

    # Sometimes it is more convenient to use wide fov to see the whole cloud from all the view points.
    # so the FOV is also tuned:
    # -- TUNE FOV, CNY,CNX:
    if run_params['IFTUNE_CAM']:
        L *= 2 # 1.6
        fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['Rsat'])))
        vis_cnx = vis_cny = int(np.floor(L / vis_pixel_footprint))
        if USESWIR:
            swir_cnx = swir_cny = int(np.floor(L / swir_pixel_footprint))

    # Update the resolution of each Imager with respect to new pixels number [nx,ny].
    # In addition we update Imager's FOV.
    vis_imager.update_sensor_size_with_number_of_pixels(vis_cnx, vis_cny)
    if USESWIR:
        swir_imager.update_sensor_size_with_number_of_pixels(swir_cnx, swir_cny)

    # not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
    # tuning is applied by the variable LOOKAT.
    LOOKAT = CENTER_OF_MEDIUM_BOTTOM
    if run_params['IFTUNE_CAM']:
        LOOKAT[2] = droplets_grid.bounding_box.zmin + 0.1 # tuning. if IFTUNE_CAM = False, just lookat the bottom
        
    # currently, all satellites lookat the same point.
    SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1, 3)

    logger.debug("CAMERA intrinsics summary")
    logger.debug(f"vis: fov = {fov}[deg], cnx = {vis_cnx}[pixels],cny ={vis_cny}[pixels]")
    if USESWIR:
        logger.debug(f"swir: fov = {fov}[deg], cnx = {swir_cnx}[pixels],cny ={swir_cny}[pixels]")

    logger.debug("Medium summary")
    logger.debug(f"nx = {nx}, ny = {ny},nz ={nz}")
    logger.debug(f"dx = {dx}, dy = {dy},dz ={dz}")
    logger.debug(f"Lx = {Lx}, Ly = {Ly},Lz ={Lz}")

    logger.debug(
        f"xmin = {droplets_grid.bounding_box.xmin}, "
        f"ymin = {droplets_grid.bounding_box.ymin}, "
        f"zmin ={droplets_grid.bounding_box.zmin}")
    logger.debug(
        f"xmax = {droplets_grid.bounding_box.xmax}, "
        f"ymax = {droplets_grid.bounding_box.ymax}, "
        f"zmax ={droplets_grid.bounding_box.zmax}")

    if run_params['DOFORWARD']:
        forward_options = run_params['forward_options']

        # ---------------------------------------------------------------
        # ---------------CREATE THE SETUP----------------------------
        # ---------------------------------------------------------------
        """
                The forward simulation will run with unity flux in the input.
                Imager.L_TOA is a scale that we need to apply on the output images.
        """

        # create CloudCT setups:
        vis_rays_per_pixel = run_params['vis_options']['rays_per_pixel']
        rigid_sampling = run_params['vis_options']['rigid_sampling'] # True of False
        vis_CloudCT_VIEWS, _ = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP, WIDEST_VIEW = run_params['WIDEST_VIEW'],
                                                    ORBIT_ALTITUDE=run_params['Rsat'],
                                                    SAT_LOOKATS=SAT_LOOKATS,
                                                    Imager_config=vis_imager_config,
                                                    imager=vis_imager,
                                                    samples_per_pixel = vis_rays_per_pixel,rigid_sampling = rigid_sampling,
                                                    APPLY_VIEW_GEOMETRIC_SCALING = run_params['APPLY_VIEW_GEOMETRIC_SCALING'],
                                                    VISSETUP=vizual_options['VISSETUP'])
        # If APPLY_VIEW_GEOMETRIC_SCALING is True:
        # Add additional scaling to the projection. It will be used to scale the gradient per view only.
                
            
        if USESWIR:
            swir_rays_per_pixel = run_params['swir_options']['rays_per_pixel']
            rigid_sampling = run_params['swir_options']['rigid_sampling'] # True of False
            swir_CloudCT_VIEWS, _ = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP, WIDEST_VIEW = run_params['WIDEST_VIEW'],
                                                         ORBIT_ALTITUDE=run_params['Rsat'],
                                                         SAT_LOOKATS=SAT_LOOKATS,
                                                         Imager_config=swir_imager_config,
                                                         imager=swir_imager,
                                                         samples_per_pixel = swir_rays_per_pixel,rigid_sampling = rigid_sampling,
                                                         APPLY_VIEW_GEOMETRIC_SCALING = run_params['APPLY_VIEW_GEOMETRIC_SCALING'],
                                                         VISSETUP=vizual_options['VISSETUP'])


        # Generate a solver array for a multi-spectral solution.
        # it is great that we can use the parallel solution of all solvers.
        # -----IMPORTANT NOTE---------
        # Right now the /numerical parameter of vis and swir are the same:

        rte_solvers = shdom.RteSolverArray()

    
        """
        If the retrieval does not assume ground truht mask, this is a place to pad the scatterer to be retrieved. 
        It must be done before the rendering simulation to avoide diviations in images (simulated by forward and on optimization rendering)
        coused by shdom processing.
        """
        if not run_params['inverse_options']['use_forward_mask']:
            # pad medium all sides by XPAD, YPAD, ZPAD:
            XPAD = run_params['inverse_options']['XPAD']
            YPAD = run_params['inverse_options']['YPAD']
            ZPAD = run_params['inverse_options']['ZPAD']
            # X DIRECTION
            values = np.zeros(XPAD)
            atmosphere.pad_scatterer(name='cloud',axis=0, right = True, values=values)
            atmosphere.pad_scatterer(name='cloud',axis=0, right = False, values=values)
            # Y DIRECTION
            values = np.zeros(YPAD)
            atmosphere.pad_scatterer(name='cloud',axis=1, right = True, values=values)
            atmosphere.pad_scatterer(name='cloud',axis=1, right = False, values=values)
            # Z DIRECTION
            values = np.zeros(ZPAD)
            atmosphere.pad_scatterer(name='cloud',axis=2, right = True, values=values)
            atmosphere.pad_scatterer(name='cloud',axis=2, right = False, values=values)
            # get the updated droplets:
            droplets = copy.deepcopy(atmosphere.get_scatterer('cloud'))

        # TODO - what if the imagers have the same central wavelengths?
        # iter 0 of wavelength is for vis
        # iter 1 of wavelength is for swir
        for wavelength, split_accuracy, solar_flux, surface_albedo in zip(wavelengths_micron,
                                                                          forward_options['split_accuracies'],
                                                                          forward_options['solar_fluxes'],
                                                                          forward_options['surface_albedos']):
            numerical_params = shdom.NumericalParameters(num_mu_bins=forward_options['num_mu'],
                                                         num_phi_bins=forward_options['num_phi'],
                                                         split_accuracy=split_accuracy,
                                                         max_total_mb=forward_options['max_total_mb'])

            scene_params = shdom.SceneParameters(wavelength=wavelength,
                                                 surface=shdom.LambertianSurface(albedo=surface_albedo),
                                                 source=shdom.SolarSource(azimuth=run_params['sun_azimuth'],
                                                                          zenith=run_params['sun_zenith'],
                                                                          flux=solar_flux))

            # ---------initilize an RteSolver object---------
            rte_solver = shdom.RteSolver(scene_params, numerical_params)
            rte_solver.set_medium(atmosphere)
            rte_solvers.add_solver(rte_solver)

        # ---------RTE SOLVE ----------------------------
        rte_solvers.solve(maxiter=forward_options['rte_solver_max_iter'])
        #rte_solvers.solve(maxiter=1)
        # -----------------------------------------------
        # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
        # -----------------------------------------------
        """
        Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
        The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
        """

        # the order of the bands is important here.
        if USESWIR:
            CloudCT_measurements = CloudCT_setup.SpaceMultiView_Measurements([vis_CloudCT_VIEWS, swir_CloudCT_VIEWS])
        else:
            CloudCT_measurements = CloudCT_setup.SpaceMultiView_Measurements([vis_CloudCT_VIEWS])

        if not run_params['USE_SIMPLE_IMAGER']:
            # If we do not use simple imager we probably use imager such that noise can be applied:
            reduce_exposure = True
            scale_ideally = False
            apply_noise = True

            # Cancel noise only for test: Than uncomment the above.
            # - reduce_exposure = False
            # - scale_ideally = True
            # - apply_noise = False
            #reduce_exposure = False
            #scale_ideally = True
            #apply_noise = False            
        else:
            # If we do not use realistic imager, we MUST use IF_SCALE_IDEALLY=True so the images will have values in
            # maximum range.
            reduce_exposure = False
            scale_ideally = True
            apply_noise = False

        CloudCT_measurements.simulate_measurements(n_jobs=run_params['n_jobs'],
                                                   rte_solvers=rte_solvers,
                                                   IF_REDUCE_EXPOSURE=reduce_exposure,
                                                   IF_SCALE_IDEALLY=scale_ideally,
                                                   IF_APPLY_NOISE=apply_noise,
                                                   IF_CALIBRATION_UNCERTAINTY = run_params['use_cal_uncertainty'],
                                                   uncertainty_options = run_params['uncertainty_options'])                                                   

                    
        
        # Calculate irradiance of the spesific wavelength:
        # use plank function:
        Cosain = np.cos(np.deg2rad((180 - run_params['sun_zenith'])))
        temp = 5900  # K

        L_TOA_vis = 6.8e-5 * 1e-9 * plank(1e-6 * np.array(CloudCT_measurements._imagers_unique_wavelengths_list[0]), temp)  # units fo W/(m^2)
        solar_flux_vis = L_TOA_vis * Cosain
        # scale the radiance:
        vis_max_radiance_list = [image.max()*solar_flux_vis for image in CloudCT_measurements._Radiances_per_imager[0]]

        if USESWIR:
            L_TOA_swir = 6.8e-5 * 1e-9 * plank(1e-6 * np.array(CloudCT_measurements._imagers_unique_wavelengths_list[1]), temp)  # units fo W/(m^2)
            solar_flux_swir = L_TOA_swir * Cosain
            # scale the radiance: 
            swir_max_radiance_list = [image.max()*solar_flux_swir for image in CloudCT_measurements._Radiances_per_imager[1]]

        else:
            swir_max_radiance_list = None
            
        # The simulate_measurements() simulate images in gray levels.
        # Now we need to save that images and be able to load that images in the inverse pipline/

        # ------------------------------------------------------------------------
        # ----------Threshold for the radiance to create a cloud mask.------------
        # ------------------------------------------------------------------------
        radiance_threshold_dict = dict()
        if(not run_params['inverse_options']['use_forward_mask']):
            # In a case of use_forward_mask = False which means that the 3D mask will be 
            # calculated by Space Carving, we will visualize the images after radiance trashold on radiances images.
            
            radiance_threshold_vis = run_params['inverse_options']['radiance_threshold_vis']
            radiance_threshold_swir = run_params['inverse_options']['radiance_threshold_swir']
            assert isinstance(radiance_threshold_vis,list) , "radiance_threshold must be a list"
            if len(radiance_threshold_vis) == 1:
                # index 0 always to vis
                radiance_threshold_dict[0] = sum(vis_imager_config)*radiance_threshold_vis
            
            else:
                radiance_threshold_dict[0] = radiance_threshold_vis
                
            if USESWIR:
                assert isinstance(radiance_threshold_swir,list) , "radiance_threshold must be a list"
                if len(radiance_threshold_vis) == 1:
                    radiance_threshold_dict[1] = sum(swir_imager_config)*radiance_threshold_swir
            
                else:
                    radiance_threshold_dict[1] = radiance_threshold_swir

        """
        Merge Shubi's update here:
        if run_params['inverse_options']['vis_radiance_threshold_path']:
        with open(run_params['inverse_options']['vis_radiance_threshold_path'], 'rb') as f:
            vis_radiance_thresholds = np.load(f)
        else:
            vis_radiance_thresholds = run_params['inverse_options']['default_radiance_threshold']
    
        if run_params['inverse_options']['swir_radiance_threshold_path']:
            with open(run_params['inverse_options']['swir_radiance_threshold_path'], 'rb') as f:
                swir_radiance_thresholds = np.load(f)
        else:
            swir_radiance_thresholds = run_params['inverse_options']['default_radiance_threshold']


        """
        # ------------------------------------------------------------------------                 
        # ----------------See the simulated images------------:
        # ------------------------------------------------------------------------
        if forward_options['SEE_IMAGES']:
            if(run_params['inverse_options']['use_forward_mask']):
                
                CloudCT_measurements.show_measurments(title_content=run_params['sun_zenith'])
                
            else:
      
                CloudCT_measurements.show_measurments(title_content=run_params['sun_zenith'], 
                                                      radiance_threshold_dict = radiance_threshold_dict)                
            plt.show()

            
        if not run_params['inverse_options']['use_forward_mask']:
            print("Calculating the mask by space curving for:")
            print("\t 1. Initialization with Monotonous profile.")
            print("\t 2. Optimization.")
            print("\t 3. Fitting of  Monotonous profile.") 
            print("\t 4. Overwrite erode_edges_init to be True to avoide large mass in edges and cloud top.") 
            
            run_params['inverse_options']['erode_edges_init'] = True
            
            # Apply Space curving:
            space_carver = shdom.SpaceCarver(CloudCT_measurements)
            if isinstance(rte_solvers, shdom.RteSolverArray):
                rte_solver = rte_solvers[0]
            else:
                rte_solver = rte_solvers
            PYTHON_SPACE_CURVE = run_params['inverse_options']['CloudCT_space_curve']
            agreement = run_params['inverse_options']['agreement']
            radiance_threshold_vis = run_params['inverse_options']['radiance_threshold_vis']
            radiance_threshold_swir = run_params['inverse_options']['radiance_threshold_swir'] 
            num_vis = len(run_params['vis_options']['true_indices'])
            num_swir = len(run_params['swir_options']['true_indices'])
            
            # incase radiance_threshold_vis is a number
            if not isinstance(radiance_threshold_vis, list):
                radiance_threshold_vis = [radiance_threshold_vis]
            # Now if it is a list but with only 1 entry
            if len(radiance_threshold_vis) == 1:
                radiance_threshold_vis *= num_vis
            
            # similarly for swir
            if not isinstance(radiance_threshold_swir, list):
                radiance_threshold_swir = [radiance_threshold_swir]    
            if len(radiance_threshold_swir) == 1:
                radiance_threshold_swir *= num_swir                
                
            thresholds = radiance_threshold_vis + radiance_threshold_swir
            curved_mask = space_carver.carve(droplets.grid, thresholds=thresholds, agreement=agreement, PYTHON_SPACE_CURVE = PYTHON_SPACE_CURVE)
            CHECK_CURVED_MASK  = False
            if CHECK_CURVED_MASK:
            # make sure thresholds are valide if it is one value or a list etc.
            # Plot 
            # show -> check - > delete:
            # visualize volume:
                try:
                    import mayavi.mlab as mlab
            
                except:
                    raise Exception("Make sure you installed mayavi")
            
                from scipy import ndimage
                
                #-------show images-
                space_carver_images = space_carver.images
                f, axarr = plt.subplots(1, len(space_carver_images), figsize=(20, 20))
                for ax, image, threshold in zip(axarr, space_carver_images, thresholds):
                    img = image.copy()
                    img[image < threshold] = 0
                    ax.imshow(img)
                    ax.invert_xaxis() 
                    ax.invert_yaxis() 
                    ax.axis('off')
                #-------show carved volume-
                MAXI = 1
                if isinstance(wavelengths_micron,list):
                    wavelength_test = wavelengths_micron[0]
                else:
                    wavelength_test = wavelengths_micron
                    
                ref_vol = droplets.get_extinction(wavelength = wavelength_test)
                ref_mask = ref_vol.data > 0.001
                ref_mask = np.multiply(ref_mask, 1, dtype= np.int16)
                show_vol = np.multiply(curved_mask.data, 1, dtype= np.int16) 
                
                mlab.figure()
                h = mlab.pipeline.scalar_field(show_vol)
                v = mlab.pipeline.volume(h,vmin=0.0,vmax=MAXI)
            
                ipw_x = mlab.pipeline.image_plane_widget(h, plane_orientation='x_axes',vmin=0.0,vmax=MAXI)
                ipw_x.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_x.ipw.texture_interpolate = False
                ipw_y = mlab.pipeline.image_plane_widget(h, plane_orientation='y_axes',vmin=0.0,vmax=MAXI)
                ipw_y.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_y.ipw.texture_interpolate = False
                ipw_z = mlab.pipeline.image_plane_widget(h, plane_orientation='z_axes',vmin=0.0,vmax=MAXI)
                ipw_z.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_z.ipw.texture_interpolate = False
            
                color_bar = mlab.colorbar(orientation='vertical', nb_labels=5)
                mlab.outline(color = (1, 1, 1))  # box around data axes
                mlab.title('calculated mask')
                # -----------------------------------------
                diff_mask = ref_mask - show_vol
                beta_test_mask = diff_mask == 1
                beta_test = np.zeros_like(ref_vol.data)
                beta_test[beta_test_mask] = ref_vol.data[beta_test_mask]
                
                mlab.figure()
                
                h = mlab.pipeline.scalar_field(diff_mask)
                v = mlab.pipeline.volume(h,vmin=-1,vmax=1)
            
                ipw_x = mlab.pipeline.image_plane_widget(h, plane_orientation='x_axes',vmin=-1,vmax=1)
                ipw_x.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_x.ipw.texture_interpolate = False
                ipw_y = mlab.pipeline.image_plane_widget(h, plane_orientation='y_axes',vmin=-1,vmax=1)
                ipw_y.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_y.ipw.texture_interpolate = False
                ipw_z = mlab.pipeline.image_plane_widget(h, plane_orientation='z_axes',vmin=-1,vmax=1)
                ipw_z.ipw.reslice_interpolate = 'nearest_neighbour'
                ipw_z.ipw.texture_interpolate = False
            
                color_bar = mlab.colorbar(orientation='vertical', nb_labels=5)
                mlab.outline(color = (1, 1, 1))  # box around data axes
                mlab.title('diff')
                # -----------------------------------------
                # -----------------------------------------
                
                mlab.show()            
                 
              

        # ---------SAVE EVERYTHING FOR THIS SETUP -------
        medium = atmosphere
        shdom.save_CloudCT_measurments_and_forward_model(directory=forward_dir, medium=medium, solver=rte_solvers,
                                                         measurements=CloudCT_measurements)

        logger.debug("Forward phase complete")


    # ---------SOLVE INVERSE ------------------------
    if run_params['DOINVERSE']:
        inverse_options = run_params['inverse_options']

        # See the simulated images:
        if inverse_options['SEE_IMAGES']:
            # load the cloudct measurments to see the rendered images:
            medium, solver, CloudCT_measurements = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
            # A CloudCT Measurements object bundles together imaging geometry and sensor measurements for later optimization            
            CloudCT_measurements.show_measurments()
            plt.show()
            
            
        # ---------what to optimize----------------------------
        run_type = inverse_options['recover_type'] if inverse_options['MICROPHYSICS'] else 'extinction'

        log_name = run_type + "_only_" + log_name_base
        log_name = run_params['Log_Prefix'] + log_name

        cmd = create_inverse_command(run_params=run_params, inverse_options=inverse_options,
                                     vizual_options=vizual_options,
                                     forward_dir=forward_dir, AirFieldFile=AirFieldFile,
                                     run_type=run_type, log_name=log_name, radiance_threshold_dict=radiance_threshold_dict,
                                     force_1d_reff = inverse_options['force_1d_reff'])

        dump_run_params(run_params=run_params, dump_dir=forward_dir)

        write_to_run_tracker(forward_dir=forward_dir, msg=[time.strftime("%d-%b-%Y-%H:%M:%S"), log_name, cmd])

        logger.debug(f'Starting Inverse: {cmd}')
        log2load = os.path.join(forward_dir,'logs',log_name+'USER_CHOOSE_TIME')
        logger.debug(f'tensorboard command: tensorboard --logdir {log2load} --bind_all')

        print('inverse command:')
        print(cmd)
        _ = subprocess.call(cmd, shell=True)

        # Time to show the results in 3D visualization:
        if inverse_options['VIS_RESULTS3D']:
            # TODO wavelength_micron=wavelengths_micron[0] is just placeholder for?
            visualize_results(forward_dir=forward_dir, log_name=log_name, wavelength_micron=wavelengths_micron[0])

        logger.debug("Inverse phase complete")
        
        
    elif run_params['DOCOSTEVAL']:
        # just calculate the cost function in different initialization modes:
        inverse_options = run_params['inverse_options']
        CloudFieldFile = run_params['CloudFieldFile']
        # ---------what perameter to initialize----------------------------
        run_type = inverse_options['recover_type'] if inverse_options['MICROPHYSICS'] else 'extinction'
        log_name = 'cost_' + log_name_base + '-' + time.strftime("%d-%b-%Y-%H:%M:%S")
        log_dir_first_part = os.path.join(forward_dir, 'logs')
        log_dir = os.path.join(log_dir_first_part, log_name)
        if not os.path.exists(log_dir_first_part):
            os.mkdir(log_dir_first_part)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            
        reff_tracker_path = os.path.join(log_dir,'reff.txt')
        lwc_tracker_path = os.path.join(log_dir,'lwc.txt')
        cost_tracker_path = os.path.join(log_dir,'cost.txt')
        
        if(os.path.exists(reff_tracker_path)):
            os.remove(reff_tracker_path)
        if(os.path.exists(lwc_tracker_path)):
            os.remove(lwc_tracker_path)
        if(os.path.exists(cost_tracker_path)):
            os.remove(cost_tracker_path)
            
        reff_list = np.linspace(3,15,16)
        lwc_list = np.linspace(0.1,1.6,16)
        
        # for rico only:
        #reff_list = np.linspace(8,20,16)
        #lwc_list = np.linspace(0.1,1.6,16)        

        for reff in reff_list:
            for lwc in lwc_list:
                logger.debug(f"{CloudFieldFile.split('/')[-1]}: (reff,lwc) =  ({reff}, {lwc})")
                with open(lwc_tracker_path, 'a') as file_object:
                    file_object.write('{}\n'.format(lwc))
                
                with open(reff_tracker_path, 'a') as file_object:
                    file_object.write('{}\n'.format(reff))
                    
                # the cost is calculated inside the ptimize_microphysics_lbfgs.py:
                run_params['inverse_options']['reff'] = reff 
                run_params['inverse_options']['lwc'] = lwc
                                
                cmd = create_inverse_command(run_params=run_params, inverse_options=inverse_options,
                                             vizual_options=vizual_options,
                                             forward_dir=forward_dir, AirFieldFile=AirFieldFile,
                                             run_type=run_type, log_name=log_name, radiance_threshold_dict=radiance_threshold_dict)                
                
                cmd = cmd + ' --calc_only_cost'    
                _ = subprocess.call(cmd, shell=True)
                
        logger.debug("Calculation of costs is complete")
                
                
        
        
        
        
    else:
        print('Forward is finished')

    logger.debug("--------------- End of Simulation ---------------")

    return #vis_max_radiance_list, swir_max_radiance_list


def write_to_run_tracker(forward_dir, msg):
    """
    TODO
    Args:
        forward_dir ():
        msg ():

    Returns:

    """
    tracker_path = os.path.join(forward_dir, 'run_tracker.csv')
    with open(tracker_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(msg)
    logger = logging.getLogger()
    logger.debug(f"Wrote to tracker{tracker_path} message - {msg}")


def visualize_results(forward_dir, log_name, wavelength_micron):
    """
    TODO
    Args:
        forward_dir ():
        log_name ():
        wavelength_micron ():

    Returns:

    """
    """
         The forward_dir id a folder that containes:
         medium, solver, measurements.
         They loaded before. To see the final state, the medium is not
         enough, the medium_estimator is needed.
         load the measurments to see the rendered images:
     """

    # what state to load? I prefere the last one!
    logs_dir = os.path.join(forward_dir, 'logs')
    logs_prefix = os.path.join(logs_dir, log_name)
    logs_files = glob.glob(logs_prefix + '-*')

    times = [i.split('{}-'.format(int(1e3 * wavelength_micron)))[-1] for i in logs_files]
    # sort the times to find the last one.
    timestamp = [time.mktime(time.strptime(i, "%d-%b-%Y-%H:%M:%S")) for i in times]
    # time.mktime(t) This is the inverse function of localtime()
    timestamp.sort()
    timestamp = [time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime(i)) for i in timestamp]
    # now, the timestamp are sorted, and I want the last stamp to visualize.
    connector = '{}-'.format(int(1e3 * wavelength_micron))
    log2load = logs_prefix + '-' + timestamp[-1]
    # print here the Final results files:
    Final_results_3Dfiles = glob.glob(log2load + '/FINAL_3D_*.mat')
    print("{} files with the results in 3D were created:".format(len(Final_results_3Dfiles)))
    for _file in Final_results_3Dfiles:
        print(_file)

    # ---------------------------------------------------------
    # Don't want to use it now, state2load = os.path.join(log2load,'final_state.ckpt')
    logger = logging.getLogger()
    logger.debug(f"use tensorboard --logdir {log2load} --bind_all")


def setup_imager(imager_options, run_params, MieTablesPath, simple_type):
    """
    TODO
    Args:
        imager_options ():
        run_params ():
        MieTablesPath ():
        simple_type ():

    Returns:

    """
    if not run_params['USE_SIMPLE_IMAGER']:
        imager = shdom.Imager.ImportConfig(file_name=imager_options['ImportConfigPath'])
        imager.set_Imager_altitude(H=run_params['Rsat'])
        imager_pixel_footprint, _ = imager.get_footprints_at_nadir()
        imager_pixel_footprint = float_round(imager_pixel_footprint)
        imager.change_temperature(imager_options['temperature'])
    else:
        # the imager is simple and will be defined here:
        simple_sensor = shdom.SensorFPA(PIXEL_SIZE=1, CHeight=100, CWidth=100)
        simple_lens = shdom.LensSimple(FOCAL_LENGTH=1, DIAMETER=1)
        # shdom.LensSimple means that the lens model is simple and without MTF considerations but still with focal
        # and diameter.

        imager = shdom.Imager(sensor=simple_sensor, lens=simple_lens)
        # you must define the spectrum for each defined imager:
        if simple_type == 'vis':
            imager.set_scene_spectrum_in_microns([0.672, 0.672])  #
        elif simple_type == 'swir':
            imager.set_scene_spectrum_in_microns([1.6, 1.6])
        else:
            print("if using simple imager, simple_type must be 'vis' or 'swir'")
            return
        imager.set_Imager_altitude(H=run_params['Rsat'])
        imager_pixel_footprint = run_params['GSD']
        imager.set_pixel_footprint(imager_pixel_footprint)

    # update solar irradince with the solar zenith angel:
    imager.update_solar_angle(run_params['sun_zenith'])
    # TODO -  use here pysat   or pyEpham package to predict the orbital position of the nadir view sattelite.
    # It will be used to set the solar zenith.

    # calculate mie tables:
    wavelegth_range_nm = imager.scene_spectrum  # in nm

    # convert nm to microns: It is must
    wavelegth_range = [float_round(1e-3 * w) for w in wavelegth_range_nm]

    """
    Check if mie tables exist, if not creat them, if yes skip it is long process.
    table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
    """

    if wavelegth_range[0] == wavelegth_range[1]:
        _ = CALC_MIE_TABLES(where_to_check_path=MieTablesPath,
                            wavelength_micron=wavelegth_range[0],
                            options=run_params['mie_options'],
                            wavelength_averaging=False)
    else:
        _ = CALC_MIE_TABLES(where_to_check_path=MieTablesPath,
                            wavelength_micron=wavelegth_range,
                            options=run_params['mie_options'],
                            wavelength_averaging=True)

    imager_config = run_params['SATS_NUMBER_SETUP'] * [False]
    for index in imager_options['true_indices']:
        imager_config[index] = True

    return imager, wavelegth_range, imager_pixel_footprint, imager_config


def dump_run_params(run_params, dump_dir):
    """
    TODO
    Args:
        run_params ():
        dump_dir ():

    Returns:

    """
    logger = logging.getLogger()
    if not os.path.exists(os.path.join(dump_dir, 'run_params_files')):
        os.mkdir(os.path.join(dump_dir, 'run_params_files'))

    run_params_file_name = os.path.join(dump_dir, 'run_params_files',
                                        'run_params_' + time.strftime("%d%m%Y_%H%M%S") + '.yaml')

    with open(run_params_file_name, 'w') as f:
        yaml.dump(run_params, f)
        logger.debug(f"Saving run params to {run_params_file_name}")


def create_inverse_command(run_params, inverse_options, vizual_options,
                           forward_dir, AirFieldFile, run_type, log_name, radiance_threshold_dict, force_1d_reff=False):
    """
    TODO
    Args:
        run_params ():
        inverse_options ():
        vizual_options ():
        forward_dir ():
        AirFieldFile ():
        run_type ():
        log_name ():

    Returns:

    """
    if(inverse_options['init'] == 'Homogeneous' ):
        INIT_USE = ' --init ' + inverse_options['init']
    elif(inverse_options['init'] == 'LesFile' ):
        INIT_USE = ' --init ' + inverse_options['init'] + ' --path ' + inverse_options['LesFile_path']
    elif(inverse_options['init'] == 'Monotonous' ): 
        INIT_USE = ' --init ' + inverse_options['init']  
    elif(inverse_options['init'] == 'FromMatFile' ): 
        INIT_USE = ' --init ' + inverse_options['init'] + ' --path ' + inverse_options['mat_path']
        
        
    GT_USE = ''
    GT_USE = GT_USE + ' --cloudct_use' if inverse_options['cloudct_use'] else GT_USE  # -----------------
    GT_USE = GT_USE + ' --add_rayleigh' if inverse_options['add_rayleigh'] and not run_params['forward_options'][
        'CENCEL_AIR'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_mask' if inverse_options['use_forward_mask'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_grid' if inverse_options['use_forward_grid'] else GT_USE
    GT_USE = GT_USE + ' --save_gt_and_carver_masks' if inverse_options['if_save_gt_and_carver_masks'] else GT_USE
    GT_USE = GT_USE + ' --save_final3d' if inverse_options['if_save_final3d'] else GT_USE
    GT_USE = GT_USE + ' --python_space_curve' if run_params['inverse_options']['CloudCT_space_curve'] else GT_USE
    GT_USE = GT_USE + ' --force_1d_reff' if force_1d_reff else GT_USE
    
    # The mie_base_path is defined at the beginning of this script.
    # (use_forward_mask, use_forward_grid, use_forward_albedo, use_forward_phase):
    # Use the ground-truth things. This is an inverse crime which is
    # useful for debugging/development.

    # The log_name defined above
    # Write intermediate TensorBoardX results into log_name.
    # The provided string is added as a comment to the specific run.

    # note that currently, the invers_dir = forward_dir.
    # In forward_dir, the forward modeling parameters are be saved.
    # If inverse_dir = forward_dir:
    # The inverse_dir directory will be used to save the optimization results and progress.

    OTHER_PARAMS = ' --input_dir ' + forward_dir + \
                   ' --log ' + log_name + \
                   ' --n_jobs ' + str(run_params['n_jobs']) + \
                   ' --loss_type ' + inverse_options['loss_type'] + \
                   ' --maxls ' + str(inverse_options['maxls']) + \
                   ' --maxiter ' + str(inverse_options['maxiter'])

    OTHER_PARAMS = OTHER_PARAMS + ' --erode_edges_init' if inverse_options['erode_edges_init'] else OTHER_PARAMS
    OTHER_PARAMS = OTHER_PARAMS + ' --precond_grad' if inverse_options['precond_grad'] else OTHER_PARAMS
    OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if inverse_options['globalopt'] else OTHER_PARAMS

    OTHER_PARAMS = OTHER_PARAMS + ' --air_path ' + AirFieldFile if not run_params['forward_options']['CENCEL_AIR'] else OTHER_PARAMS
    OTHER_PARAMS += ' --smoothness_const ' + str(inverse_options['reff_smoothness_const'])
    
    if(not run_params['forward_options']['CENCEL_AIR']):
        OTHER_PARAMS = OTHER_PARAMS + ' --air_max_alt ' + str(run_params['forward_options']['air_max_alt'])
        OTHER_PARAMS = OTHER_PARAMS + ' --air_num_points ' + str(run_params['forward_options']['air_num_points'])

    if not run_params['inverse_options']['use_forward_mask']:
        """
        Her we must use radiance treshold:
        Threshold for the radiance to create a cloud mask.
        Threshold is either a scalar or a list of length of measurements.
        """
        radiance_thresholds = list(chain(*radiance_threshold_dict.values())) # Concatenating dictionary value lists. 
            
        OTHER_PARAMS = OTHER_PARAMS + ' --radiance_threshold ' + " ".join(
            map(str, radiance_thresholds))
        
        
    if inverse_options['MICROPHYSICS']:
        # -----------------------------------------------
        # ---------SOLVE for lwc only  ------------------
        # -----------------------------------------------
        if run_type == 'lwc':
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and variance, Hence,
               the albedo should be known.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """

            GT_USE += ' --use_forward_reff'
            GT_USE += ' --use_forward_veff'
            OTHER_PARAMS += ' --lwc ' + str(inverse_options['lwc'])

        # -----------------------------------------------
        # ---------SOLVE for reff only  ------------------
        # -----------------------------------------------
        elif run_type == 'reff':
            """
            Estimate reff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective variance and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
            """

            GT_USE += ' --use_forward_lwc'
            GT_USE += ' --use_forward_veff' 
            OTHER_PARAMS += ' --reff ' + str(inverse_options['reff'])

        # -----------------------------------------------
        # ---------SOLVE for veff only  ------------------
        # -----------------------------------------------
        elif run_type == 'veff':
            """
            Estimate veff with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. ground-truth effective radius and lwc.
            4. rayleigh scattering (add_rayleigh)
            5. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """

            GT_USE += ' --use_forward_lwc'
            GT_USE += ' --use_forward_reff'
            OTHER_PARAMS += ' --veff ' + str(inverse_options['veff'])

        # -----------------------------------------------
        # ---------SOLVE for lwc and reff  ------------------
        # -----------------------------------------------
        elif run_type == 'reff_and_lwc':
            """
            Estimate lwc with (what is known?):
            1. ground truth phase function (use_forward_phase, with mie_base_path)
            2. grid (use_forward_grid)
            3. rayleigh scattering (add_rayleigh)
            4. cloud mask (use_forward_mask) or not (when use_forward_mask = False).

            """
            if run_params['inverse_options']['use_const_veff']:
                GT_USE += ' --const_veff --veff '+ str(run_params['inverse_options']['veff']) 
            else:
                GT_USE += ' --use_forward_veff' 
                
            GT_USE += ' --lwc_scaling ' + str(inverse_options['lwc_scaling_val'])
            GT_USE += ' --reff_scaling ' + str(inverse_options['reff_scaling_val'])
            OTHER_PARAMS += ' --reff ' + str(inverse_options['reff'])
            OTHER_PARAMS += ' --lwc ' + str(inverse_options['lwc'])
    # -----------------------------------------------
    # ---------SOLVE for extinction only  -----------
    # -----------------------------------------------
    else:
        """
        Estimate extinction with (what is known?):
        1. ground truth phase function (use_forward_phase, with mie_base_path)
        2. grid (use_forward_grid)
        3. cloud mask (use_forward_mask) or not (when use_forward_mask = False).
        4. known albedo (use_forward_albedo)
        5. rayleigh scattering (add_rayleigh)

        """

        # First of all we don't neet the CloudCT measurments type here sinse the extinction optimization does not
        # support inverse with more than one wavelength.
        GT_USE = GT_USE.replace('--cloudct_use', '') # it will erase this flag either it is on or off.
        # Convert the CloudCT_measurements to regular measurements and save it: 
        medium, solver, CloudCT_measurements = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
        measurements = CloudCT_measurements.convert2regular_measurements()
        shdom.save_forward_model(forward_dir, medium, solver, measurements)
        #--------------------------------------------------------------------------
        
        GT_USE += ' --use_forward_albedo'
        GT_USE += ' --use_forward_phase'
        OTHER_PARAMS += ' --extinction ' + str(inverse_options['extinction'])

    optimizer_path = inverse_options['microphysics_optimizer'] if inverse_options['MICROPHYSICS'] else \
        inverse_options[
            'extinction_optimizer']

    # We have: ground_truth, rte_solver, measurements.

    cmd = 'python ' + \
          os.path.join(inverse_options['scripts_path'], optimizer_path) + \
          OTHER_PARAMS + \
          GT_USE + \
          INIT_USE

    return cmd


def create_and_configer_logger(log_name):
    """
    TODO
    Args:
        log_name ():

    Returns:

    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.DEBUG,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger


def load_run_params(params_path):
    """
    TODO
    Args:
        params_path ():

    Returns:

    """
    # Load run parameters
    params_file_path = params_path
    with open(params_file_path, 'r') as f:
        run_params = yaml.full_load(f)

    logger = logging.getLogger(__name__)

    logger.debug(f"loading params from {params_file_path}")
    logger.debug(f"running with params:{run_params}")
    return run_params


def compare_forward_models(path1,path2):
    pad = 0.01
    from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

    file = open(path1, 'rb')
    data = file.read()
    CloudCT_measurments1 = pickle.loads(data)
    
    head_tail = os.path.split(path2) 
    file_name = head_tail[1]
    if 'mat' in file_name:
        CloudCT_measurments2 = sio.loadmat(file_name)['images']#[0]
    else:
        
        file = open(path2, 'rb')
        data = file.read()
        CloudCT_measurments2 = pickle.loads(data)
    
    setup =  CloudCT_measurments1.setup
    pad = 0.01 # colorbar pad
    i = 0
    for imager_index in CloudCT_measurments1.images.keys():
        
        names = setup[0].projection_names
        images1 = CloudCT_measurments1.images[imager_index]
        #images2 = CloudCT_measurments2.images[imager_index]
        
        #for view_index, (img1, img2) in enumerate(zip(images1, images2)):
        for view_index, img1 in enumerate(images1):
            name = view_index
            fig, ax = plt.subplots(1, 3, figsize=(20, 20))
            img2 = CloudCT_measurments2[imager_index*8 + view_index,...]
            
            MAXI = max(np.max(img1), np.max(img2))
            MINI = min(np.min(img1), np.min(img2))
            im1 = ax[0].imshow(img1,cmap='jet',vmin=MINI, vmax=MAXI)
            ax[0].set_title("{}_{}_1".format(name,i))
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=pad)
            plt.colorbar(im1, cax=cax)        
            
            im2 = ax[1].imshow(img2,cmap='jet',vmin=MINI, vmax=MAXI)
            ax[1].set_title("{}_{}_2".format(name,i))
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=pad)
            plt.colorbar(im2, cax=cax)        
            
            im3 = ax[2].imshow(img1-img2,cmap='jet')
            ax[2].set_title("diff.")
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=pad)
            plt.colorbar(im3, cax=cax) 
        
    plt.show()

def CALCULATE_COST_ON_MANY_CLOUDS():
    """
    Makse sure that the run_params has:
    DOFORWARD: True
    DOINVERSE: False
    DOCOSTEVAL: True 
    
    And main has the CloudFieldFile as input parameter.
    
    """
    # prepare clouds for cost evaluation:
    CloudFieldFiles = []
    # CloudFieldFiles.append('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_22x27x49_23040.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_24x22x21_43200.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_35x28x54_55080.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36x31x55_53760.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_13x25x36_28440.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36000_39x44x30_4821') 

    for CloudFieldFile in CloudFieldFiles:
        main(CloudFieldFile)    


def run_many_cases():
    
    CloudFieldFiles = []
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_22x27x49_23040.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_24x22x21_43200.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_35x28x54_55080.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36x31x55_53760.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_13x25x36_28440.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36000_39x44x30_4821') 

    Init_dict = {}
    Init_dict['23040'] = {'lwc':0.5,'reff':7.8}
    Init_dict['43200'] = {'lwc':1.3,'reff':10.2}
    Init_dict['55080'] = {'lwc':0.8,'reff':11.8}
    
    Init_dict['53760'] = {'lwc':0.8,'reff':11.8}
    Init_dict['28440'] = {'lwc':0.8,'reff':11.0}
    Init_dict['4821'] =  {'lwc':1.6,'reff':12.6}
    
    Prefix = "Init_same_as_swir_vis_But_with_swir_"
    logger = create_and_configer_logger(log_name='run_tracker.log')

    for CloudFieldFile in CloudFieldFiles:
        CloudFieldName = CloudFieldFile.split('/')[-1].split('.')[0]
        CloudFieldName = CloudFieldName.split('_')[-1]
        
        logger.debug(CloudFieldName)
        logger.debug('init lwc {}, init reff {}'.format(Init_dict[str(CloudFieldName)]['lwc'],
                                                 Init_dict[str(CloudFieldName)]['reff']))
        logger.debug(10*'-')
    
        main(CloudFieldFile = CloudFieldFile, Init_dict = Init_dict[str(CloudFieldName)], Prefix = Prefix, logger=logger)
    
    # -------------------------------
    
def SHOW_INIT_PROFILES(N=16):
    SHOW_FITTING_ERROR = False
    from mpl_toolkits import mplot3d
    from collections import namedtuple
    
    #PATH_TO_LOOK_FOR_COST_LOGS = '../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_620-670nm_active_sats_10_GSD_20m_and_SWIR_1628-1658nm_active_sats_10_GSD_50m_LES_cloud_field_BOMEX/logs/'
    PATH_TO_LOOK_FOR_COST_LOGS = '../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_620-670nm_active_sats_10_GSD_20m_LES_cloud_field_BOMEX/logs/'
    
    
    TEMPLATE = 'cost_reff_smoothness_with_0.0prec_lwc_10_prec_reff_0.1_'
    
    # prepare clouds for cost evaluation:
    CloudFieldFiles = []
    #CloudFieldFiles.append('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_22x27x49_23040.txt')
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_24x22x21_43200.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_35x28x54_55080.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36x31x55_53760.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_13x25x36_28440.txt') 
    CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36000_39x44x30_4821') 

    for CloudFieldFile in CloudFieldFiles:
        CloudFieldName = CloudFieldFile.split('/')[-1]
        # the date of the log is still missing here.
        FULL_PATH_WITHOUT_DATE = os.path.join(PATH_TO_LOOK_FOR_COST_LOGS,TEMPLATE+CloudFieldName+'*')
        logs_files_of_certain_cloud = glob.glob(FULL_PATH_WITHOUT_DATE)
        if len(logs_files_of_certain_cloud) > 1:
            print("There are few cost logs of cloud {}".format(CloudFieldName))
            print('The function is taking the last one in list. Make sure this is the updated one or erase the old logs.')
            log_dir = logs_files_of_certain_cloud[-1]
        else:
            log_dir = logs_files_of_certain_cloud[0]
            
        # -----------------------------------------------------------------------
        # ---------------START LOADIN AND PROCESSING-----------------------------
        # -----------------------------------------------------------------------
        reff = np.loadtxt(os.path.join(log_dir,'reff.txt'))
        lwc = np.loadtxt(os.path.join(log_dir,'lwc.txt'))
        cost = np.loadtxt(os.path.join(log_dir,'cost.txt'))
        
        
        REFF = reff.reshape(N,N)
        LWC = lwc.reshape(N,N)
        COST = cost.reshape(N,N)
        
        if(SHOW_FITTING_ERROR):
            
            fig = plt.figure(figsize=(12,8))
            ax = plt.axes(projection='3d')
            ax.plot_surface(REFF, LWC, COST, rstride=1, cstride=1,
                            cmap='jet', edgecolor='none')
            
            ax.set_title(CloudFieldName.replace('.txt',''))
            ax.view_init(elev=21., azim=28.)
            
            plt.xlabel('reff', fontsize=18)
            plt.ylabel('lwc', fontsize=18)
        
        min_index = np.argmin(cost)
        print("{}:".format(CloudFieldName))
        print("Best point is {} at (reff, lwc) = ({}, {})".format(cost[min_index], reff[min_index],lwc[min_index]))      
        alfa_reff = float_round(reff[min_index])
        alfa_lwc = float_round(lwc[min_index] )     
        # -----------------------------------------------------------------------
        # ---------------SHOW THE PROFILES TO EVALUATE---------------------------
        # -----------------------------------------------------------------------  
        gt_droplets = shdom.MicrophysicalScatterer()
        gt_droplets.load_from_csv(CloudFieldFile)
        
        # --------- mimic the init with optimal parameters:
        lwc_grid = gt_droplets.lwc.grid
        reff_grid = gt_droplets.reff.grid
        veff_grid = gt_droplets.veff.grid
        
        Arg = namedtuple('Arg',
                ['lwc',
               'reff',
               'veff'])
        
        arg = Arg( 
               lwc=lwc[min_index], 
               reff=reff[min_index], 
               veff=0.1, 
               )
        MonotonousGenerator = shdom.generate.Monotonous(arg)
        reff = MonotonousGenerator.get_reff(grid = reff_grid, min_reff = 2.5)
        if veff_grid.type == 'Homogeneous':
            veff = gt_droplets.veff
        else:
            veff = MonotonousGenerator.get_veff(grid = veff_grid)
        lwc =  MonotonousGenerator.get_lwc(grid=lwc_grid, min_lwc= 1.5e-5)
        optimat_droplets = shdom.MicrophysicalScatterer(lwc=lwc,reff=reff,veff=veff)
        
        fig, ax = plt.subplots(1,2,figsize=(12,20))
        
        title = '{} \n'.format(CloudFieldName.replace('.txt','')) + \
            r'$\alpha_{reff}$ = '+'{}\n'.format(alfa_reff) + \
            r'$\alpha_{lwc}$ = '+'{}'.format(alfa_lwc) 
        
        fig.suptitle(title, fontsize=14)
        
        mask = gt_droplets.get_mask(threshold=0.0).data
        
        for i, parameter_name in enumerate(['lwc','reff']):
            gt_param = getattr(gt_droplets,parameter_name).data
            gt_param[np.bitwise_not(mask)] = np.nan
            gt_param = np.nan_to_num(np.nanmean(gt_param,axis=(0,1)))  
            
            optimal_param = getattr(optimat_droplets,parameter_name).data
            optimal_param[np.bitwise_not(mask)] = np.nan
            optimal_param = np.nan_to_num(np.nanmean(optimal_param,axis=(0,1)))              
                        
            ax[i].set_title('{}'.format(parameter_name), fontsize=16)
            ax[i].plot(optimal_param, gt_droplets.grid.z, label='Optimal profile')
            ax[i].plot(gt_param, gt_droplets.grid.z, label='True profile')
            ax[i].legend()
            ax[i].set_ylabel('Altitude [km]', fontsize=14)            
    
    plt.show()
    # ------------------------------- 
    
    
if __name__ == '__main__':
    # run_many_cases()
    CALCULATE_COST_ON_MANY_CLOUDS()
    # #main()
    # CloudFieldFiles = []
    # #CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_22x27x49_23040.txt')
    # CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_35x28x54_55080.txt')
    # CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36x31x55_53760.txt')
    # CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_13x25x36_28440.txt')
    # CloudFieldFiles.append('../synthetic_cloud_fields/wiz/BOMEX_36000_39x44x30_4821')
    #
    # mat_paths = []
    # # must be in the same order as above:
    # #mat_paths.append('../CloudCT_experiments/noisy_init/BOMOX_23040_10_noise')
    # mat_paths.append('../CloudCT_experiments/noisy_init/BOMOX_55080_10_noise')
    # mat_paths.append('../CloudCT_experiments/noisy_init/BOMOX_53760_10_noise')
    # mat_paths.append('../CloudCT_experiments/noisy_init/BOMOX_28440_10_noise')
    # mat_paths.append('../CloudCT_experiments/noisy_init/BOMOX_4821_10_noise')
    #
    # Prefix = "Init_with_noisy_gt_"
    # init = 'FromMatFile'
    #
    # for index, CloudFieldFile in enumerate(CloudFieldFiles):
    #     CloudFieldName = CloudFieldFile.split('/')[-1].split('.')[0]
    #     CloudFieldName = CloudFieldName.split('_')[-1]
    #     mat_path = mat_paths[index]
    #     print(CloudFieldName)
    #     print(mat_path)
    #
    #     print(10*'-')
    #
    #     main(CloudFieldFile = CloudFieldFile, init = init, Prefix = Prefix, mat_path = mat_path)

        
    # -------------------------------
    if(0):
        #path1 = "/home/vhold/CloudCT/pyshdom/CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_620-670nm_active_sats_10_GSD_20m_and_SWIR_1628-1658nm_active_sats_10_GSD_50m_LES_cloud_field_BOMEX/cloudct_measurements"
        path2 = "/home/vhold/CloudCT/pyshdom/CloudCT_scripts/133_state_images.mat"
        path1 = "/home/vhold/CloudCT/pyshdom/CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_672-672nm_active_sats_10_GSD_20m_and_SWIR_1600-1600nm_active_sats_10_GSD_20m_LES_cloud_field_BOMEX/cloudct_measurements"
        compare_forward_models(path1,path2)
        
    if(0):
        forward_dir = '/home/vhold/CloudCT/pyshdom/CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_620-670nm_active_sats_10_GSD_20m_and_SWIR_1628-1658nm_active_sats_10_GSD_50m_LES_cloud_field_rico32x37x26.txt'
        medium, solver, CloudCT_measurements = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
        
        CloudCT_measurements.simulate_measurements(n_jobs=run_params['n_jobs'],
                                                   rte_solvers=solver,
                                                   IF_REDUCE_EXPOSURE=reduce_exposure,
                                                   IF_SCALE_IDEALLY=scale_ideally,
                                                   IF_APPLY_NOISE=apply_noise)        