import csv
import logging

# import mayavi.mlab as mlab
import matplotlib.pyplot as plt
# importing functools for reduce()
# importing operator for operator functions
import yaml

from shdom import CloudCT_setup
from shdom.CloudCT_Utils import *


def main():
    logger = create_and_configer_logger(log_name='run_tracker.log')

    run_params = load_run_params(params_path="run_params.yaml")

    mie_options = run_params['mie_options']  # mie params

    wavelengths_micron = run_params['wavelengths_micron']

    SATS_NUMBER_SETUP = run_params['SATS_NUMBER_SETUP']
    SATS_NUMBER_INVERSE = run_params['SATS_NUMBER_INVERSE']

    # where different imagers are located:
    vis_imager_config = SATS_NUMBER_SETUP * [True]
    swir_imager_config = SATS_NUMBER_SETUP * [False]
    swir_imager_config[4], swir_imager_config[5] = True, True

    """
        Here load the imagers, the imagers dictates the spectral bands of the rte solver and rendering.
        Since the spectrum in this script referes to narrow bands, the nerrow bands will be treated in the following maner:
        We will use the wavelength averaging of shdom. It averages scattering properties over the wavelength band.
        Be carfule, the wavelengths in Imager methods are in nm. Pyshdom the wavelength are usualy in microns.

        THe imagers aslo dictate the ground spatial resolution (GSD).
    """

    if not run_params['USE_SIMPLE_IMAGER']:

        # load Imager at VIS:
        vis_imager = shdom.Imager.ImportConfig(file_name='../notebooks/Gecko_config.json')
        # load Imager at SWIR:
        swir_imager = shdom.Imager.ImportConfig(file_name='../notebooks/Hypothetic_SWIR_camera_config.json')
    else:
        # the imager is simple it will be defined here:
        simple_sensor = shdom.SensorFPA(PIXEL_SIZE=1, CHeight=100, CWidth=100)
        simple_lens = shdom.LensSimple(FOCAL_LENGTH=1, DIAMETER=1)
        # shdom.LensSimple means that the lens model is simlpe and without MTF considerations but still with focal and diameter.

        swir_imager = shdom.Imager(sensor=simple_sensor, lens=simple_lens)
        vis_imager = shdom.Imager(sensor=simple_sensor, lens=simple_lens)
        # you must define the spectrum for each defined imager:
        swir_imager.set_scene_spectrum_in_microns([1.6, 1.6])
        vis_imager.set_scene_spectrum_in_microns([0.672, 0.672])

    Rsat = run_params['Rsat']

    # set the nadir view altitude:
    vis_imager.set_Imager_altitude(H=Rsat)
    swir_imager.set_Imager_altitude(H=Rsat)

    if not run_params['USE_SIMPLE_IMAGER']:
        # the following parametere refers only to nadir view.
        vis_pixel_footprint, _ = vis_imager.get_footprints_at_nadir()
        swir_pixel_footprint, _ = swir_imager.get_footprints_at_nadir()

        vis_pixel_footprint = float_round(vis_pixel_footprint)
        swir_pixel_footprint = float_round(swir_pixel_footprint)

    else:
        # set required pixel footprint:
        vis_pixel_footprint = 0.02  # km
        swir_pixel_footprint = 0.02  # km
        vis_imager.set_pixel_footprint(vis_pixel_footprint)
        swir_imager.set_pixel_footprint(swir_pixel_footprint)

    # play with temperatures:
    if not run_params['USE_SIMPLE_IMAGER']:
        # if we want to use the simple imager, the temperature does not make any differences.
        swir_imager.change_temperature(run_params['imager_temperature'])
        vis_imager.change_temperature(run_params['imager_temperature'])

    # update solar irradince with the solar zenith angel:
    vis_imager.update_solar_angle(run_params['sun_zenith'])
    swir_imager.update_solar_angle(run_params['sun_zenith'])
    # TODO -  use here pysat   or pyEpham package to predict the orbital position of the nadir view sattelite.
    # It will be used to set the solar zenith.

    # calculate mie tables:
    vis_wavelegth_range_nm = vis_imager.scene_spectrum  # in nm
    swir_wavelegth_range_nm = swir_imager.scene_spectrum  # in nm

    # convert nm to microns: It is must
    vis_wavelegth_range = [float_round(1e-3 * w) for w in vis_wavelegth_range_nm]
    swir_wavelegth_range = [float_round(1e-3 * w) for w in swir_wavelegth_range_nm]

    # central wavelengths.
    vis_centeral_wavelength = vis_imager.centeral_wavelength_in_microns
    swir_centeral_wavelength = swir_imager.centeral_wavelength_in_microns
    wavelengths_micron = [vis_centeral_wavelength, swir_centeral_wavelength]
    # wavelengths_micron will hold the vis swir wavelengths. It will be convinient to use the
    # wavelengths_micron in some loops.

    """
    Check if mie tables exist, if not creat them, if yes skip it is long process.
    table file name example: mie_tables/polydisperse/Water_<1000*wavelength_micron>nm.scat
    """
    MieTablesPath = os.path.abspath("./mie_tables")
    if vis_wavelegth_range[0] == vis_wavelegth_range[1]:

        vis_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                            vis_wavelegth_range[0], mie_options, wavelength_averaging=False)
    else:

        vis_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                            vis_wavelegth_range, mie_options, wavelength_averaging=True)

    if swir_wavelegth_range[0] == swir_wavelegth_range[1]:

        swir_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                             swir_wavelegth_range[0], mie_options, wavelength_averaging=False)

    else:

        swir_mie_base_path = CALC_MIE_TABLES(MieTablesPath,
                                             swir_wavelegth_range, mie_options, wavelength_averaging=True)

    # where to save the forward outputs:
    forward_dir = f'./experiments/VIS_SWIR_NARROW_BANDS_VIS_{int(1e3 * vis_wavelegth_range[0])}-' \
                  f'{int(1e3 * vis_wavelegth_range[1])}nm_active_sats_{SATS_NUMBER_SETUP}_' \
                  f'GSD_{int(1e3 * vis_pixel_footprint)}m_and' \
                  f'_SWIR__{int(1e3 * swir_wavelegth_range[0])}-{int(1e3 * swir_wavelegth_range[1])}' \
                  f'nm_active_sats_{SATS_NUMBER_SETUP}_GSD_{int(1e3 * swir_pixel_footprint)}m' \
                  f'_LES_cloud_field_rico_LES_cloud_field_rico'

    # inverse_dir, where to save evrerything that is related to invers model:
    inverse_dir = forward_dir
    log_name_base = f'active_sats_{SATS_NUMBER_SETUP}_easiest_rico32x37x26'
    # Write intermediate TensorBoardX results into log_name.
    # The provided string is added as a comment to the specific run.

    viz_options = run_params['viz_options']  # visualization params
    # -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
    # Path to csv file which contains temperature measurements or None if the atmosphere will not consider any air.
    AirFieldFile = run_params['AirFieldFile'] if not viz_options['CENCEL_AIR'] else None

    if not run_params['USE_SIMPLE_IMAGER']:
        # If we do not use simple imager we probably use imager with a band so:
        wavelength_averaging = True
    else:
        wavelength_averaging = False

    atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile=run_params['CloudFieldFile'],
                                              AirFieldFile=AirFieldFile,
                                              MieTablesPath=MieTablesPath,
                                              wavelengths_micron=wavelengths_micron,
                                              wavelength_averaging=wavelength_averaging)

    droplets = atmosphere.get_scatterer('cloud')

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
    PIXEL_FOOTPRINT = run_params['GSD']  # km
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
    vis_cnx = vis_cny = int(np.floor(L / vis_pixel_footprint))
    swir_cnx = swir_cny = int(np.floor(L / swir_pixel_footprint))

    CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

    # Sometimes it is more convenient to use wide fov to see the whole cloud from all the view points.
    # so the FOV is also tuned:
    # -- TUNE FOV, CNY,CNX:
    if run_params['IFTUNE_CAM']:
        L = 1.5 * L
        fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
        vis_cnx = vis_cny = int(np.floor(L / vis_pixel_footprint))
        swir_cnx = swir_cny = int(np.floor(L / swir_pixel_footprint))

    # Update the resolution of each Imager with respect to new pixels number [nx,ny].
    # In addition we update Imager's FOV.
    vis_imager.update_sensor_size_with_number_of_pixels(vis_cnx, vis_cny)
    swir_imager.update_sensor_size_with_number_of_pixels(swir_cnx, swir_cny)

    # not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
    # tuning is applied by the variable LOOKAT.
    LOOKAT = CENTER_OF_MEDIUM_BOTTOM
    if run_params['IFTUNE_CAM']:
        LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

    # currently, all satellites lookat the same point.
    SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1, 3)

    logger.debug("CAMERA intrinsics summary")
    logger.debug(f"vis: fov = {fov}[deg], cnx = {vis_cnx}[pixels],cny ={vis_cny}[pixels]")
    logger.debug(f"swir: fov = {fov}[deg], cnx = {vis_cnx}[pixels],cny ={vis_cny}[pixels]")

    logger.debug("Medium summary")
    logger.debug(f"nx = {nx}, ny = {ny},nz ={nz}")
    logger.debug(f"dx = {dx}, dy = {dy},dz ={dz}")
    logger.debug(f"Lx = {Lx}, Ly = {Ly},Lz ={Lz}")

    logger.debug(
        f"xmin = {droplets_grid.bounding_box.xmin}, ymin = {droplets_grid.bounding_box.ymin},zmin ={droplets_grid.bounding_box.zmin}")
    logger.debug(
        f"xmax = {droplets_grid.bounding_box.xmax}, ymax = {droplets_grid.bounding_box.ymax},zmax ={droplets_grid.bounding_box.zmax}")

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
        vis_CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP,
                                                                        ORBIT_ALTITUDE=Rsat,
                                                                        SAT_LOOKATS=SAT_LOOKATS,
                                                                        Imager_config=vis_imager_config,
                                                                        imager=vis_imager,
                                                                        VISSETUP=viz_options['VISSETUP'])

        swir_CloudCT_VIEWS, near_nadir_view_index = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP,
                                                                         ORBIT_ALTITUDE=Rsat,
                                                                         SAT_LOOKATS=SAT_LOOKATS,
                                                                         Imager_config=swir_imager_config,
                                                                         imager=swir_imager,
                                                                         VISSETUP=viz_options['VISSETUP'])

        # Generate a solver array for a multispectral solution.
        # it is great that we can use the parallel solution of all solvers.
        # -----IMPORTANT NOTE---------
        # Rigth now the /numerical parameter of vis and swir are the same:

        rte_solvers = shdom.RteSolverArray()

        # TODO - what if the imagers have the same central wavelenghts?
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

        # -----------------------------------------------
        # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
        # -----------------------------------------------
        """
        Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
        The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
        """

        # the order of the bands is important here.
        CloudCT_measurements = CloudCT_setup.SpaceMultiView_Measurements([vis_CloudCT_VIEWS, swir_CloudCT_VIEWS])

        if not run_params['USE_SIMPLE_IMAGER']:
            # If we do not use simple imager we probably use imager such that noise can be applied:
            CloudCT_measurements.simulate_measurements(n_jobs=run_params['n_jobs'],
                                                       rte_solvers=rte_solvers,
                                                       IF_REDUCE_EXPOSURE=True,
                                                      IF_SCALE_IDEALLY=False,
                                                       IF_APPLY_NOISE=True)
        else:
            # If we do not use realistic imager, we MUST use IF_SCALE_IDEALLY=True so the images will have values in maximum range.
            CloudCT_measurements.simulate_measurements(n_jobs=run_params['n_jobs'],
                                                       rte_solvers=rte_solvers,
                                                       IF_REDUCE_EXPOSURE=False,
                                                      IF_SCALE_IDEALLY=True,
                                                       IF_APPLY_NOISE=False)



        # The simulate_measurements() simulate images in gray levels.
        # Now we need to save that images and be able to load that images in the inverse pipline/

        # See the simulated images:
        if forward_options['SEE_IMAGES']:
            CloudCT_measurements.show_measurments()
            plt.show()

        # ---------SAVE EVERYTHING FOR THIS SETUP -------
        medium = atmosphere
        shdom.save_CloudCT_measurments_and_forward_model(directory=forward_dir, medium=medium, solver=rte_solvers,
                                                         measurements=CloudCT_measurements)

        logger.debug('DONE forward simulation')

    # ---------SOLVE INVERSE ------------------------
    if run_params['DOINVERSE']:
        inverse_options = run_params['inverse_options']

        # load the cloudct measurments to see the rendered images:
        medium, solver, CloudCT_measurements = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
        # A CloudCT Measurements object bundles together imaging geometry and sensor measurements for later optimization

        # See the simulated images:
        if inverse_options['SEE_IMAGES']:
            CloudCT_measurements.show_measurments()
            plt.show()

        # ---------what to optimize----------------------------
        run_type = inverse_options['recover_type'] if inverse_options['MICROPHYSICS'] else 'extinction'

        log_name = run_type + "_only_" + log_name_base

        cmd = create_inverse_command(run_params=run_params, inverse_options=inverse_options, viz_options=viz_options,
                                     forward_dir=forward_dir, AirFieldFile=AirFieldFile,
                                     run_type=run_type, log_name=log_name)
        logger.debug(f'inverse command is {cmd}')

        dump_run_params(run_params=run_params, dir=forward_dir)

        with open(os.path.join(forward_dir, 'run_tracker.csv'), 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([time.strftime("%d-%b-%Y-%H:%M:%S"), log_name, cmd])

        optimize1 = subprocess.call(cmd, shell=True)

        # Time to show the results in 3D visualization:
        if inverse_options['VIS_RESULTS3D']:
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

            logger.debug(f"use tensorboard --logdir {log2load} --bind_all")

        logger.debug("done inverse")

        # This distribution can be excellently approximated by a Gaussian
        # distribution having this expectation and variance, for nphoto
        # electr > 10, which is
        # typically the case for cameras.


def dump_run_params(run_params, dir):
    logger = logging.getLogger()
    if not os.path.exists(os.path.join(dir, 'run_params_files')):
        os.mkdir(os.path.join(dir, 'run_params_files'))

    run_params_file_name = os.path.join(dir, 'run_params_files',
                                        'run_params_' + time.strftime("%d%m%Y_%H%M%S") + '.yaml')

    with open(run_params_file_name, 'w') as f:
        yaml.dump(run_params, f)
        logger.debug(f"Saving run params to {run_params_file_name}")


def create_inverse_command(run_params, inverse_options, viz_options,
                           forward_dir, AirFieldFile, run_type, log_name):

    INIT_USE = ' --init ' + inverse_options['init']

    GT_USE = ''
    GT_USE = GT_USE + ' --cloudct_use' if inverse_options['cloudct_use'] else GT_USE  # -----------------
    GT_USE = GT_USE + ' --add_rayleigh' if inverse_options['add_rayleigh'] and not viz_options[
        'CENCEL_AIR'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_mask' if inverse_options['use_forward_mask'] else GT_USE
    GT_USE = GT_USE + ' --use_forward_grid' if inverse_options['use_forward_grid'] else GT_USE
    GT_USE = GT_USE + ' --save_gt_and_carver_masks' if inverse_options['if_save_gt_and_carver_masks'] else GT_USE
    GT_USE = GT_USE + ' --save_final3d' if inverse_options['if_save_final3d'] else GT_USE

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

    OTHER_PARAMS = OTHER_PARAMS + ' --globalopt' if inverse_options['globalopt'] else OTHER_PARAMS

    OTHER_PARAMS = OTHER_PARAMS + ' --air_path ' + AirFieldFile if not viz_options['CENCEL_AIR'] else OTHER_PARAMS

    OTHER_PARAMS = OTHER_PARAMS + ' --radiance_threshold ' + " ".join(
        map(str, run_params['radiance_threshold'])) if run_type != 'reff_and_lwc' else OTHER_PARAMS

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
    # Load run parameters
    params_file_path = params_path
    with open(params_file_path, 'r') as f:
        run_params = yaml.full_load(f)

    logger = logging.getLogger(__name__)

    logger.debug(f"loading params from {params_file_path}")
    logger.debug(f"running with params:{run_params}")
    return run_params


if __name__ == '__main__':
    main()
