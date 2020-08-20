import csv
import logging
from shutil import copyfile

import matplotlib.pyplot as plt
import yaml

from shdom import CloudCT_setup, plank
from shdom.CloudCT_Utils import *
import concurrent.futures

def main(cloud_indices):
    logger = create_and_configer_logger(log_name='run_tracker.log')
    logger.debug("--------------- New Simulation ---------------")

    run_params = load_run_params(params_path="run_params_cloud_ct_nn_test.yaml")
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

    for cloud_index in cloud_indices:
        start_time = time.time()
        # where to save the forward outputs:
        if USESWIR:
            forward_dir = f'../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_{int(1e3 * vis_wavelength_range[0])}-' \
                          f'{int(1e3 * vis_wavelength_range[1])}nm_active_sats_{SATS_NUMBER_SETUP}_' \
                          f'GSD_{int(1e3 * vis_pixel_footprint)}m_and' \
                          f'_SWIR_{int(1e3 * swir_wavelength_range[0])}-{int(1e3 * swir_wavelength_range[1])}' \
                          f'nm_active_sats_{SATS_NUMBER_SETUP}_GSD_{int(1e3 * swir_pixel_footprint)}m' \
                          f'_LES_cloud_field_rico'
        else:
            forward_dir = f'../CloudCT_experiments/cloud{cloud_index}'

        # inverse_dir, where to save everything that is related to invers model:
        inverse_dir = forward_dir  # TODO not in use
        log_name_base = f'active_sats_{SATS_NUMBER_SETUP}_easiest_rico32x37x26'
        # Write intermediate TensorBoardX results into log_name.
        # The provided string is added as a comment to the specific run.

        vizual_options = run_params['vizual_options']  # visualization params
        # -------------LOAD SOME MEDIUM TO RECONSTRUCT--------------------------------
        # Path to csv file which contains temperature measurements or None if the atmosphere will not consider any air.
        AirFieldFile = run_params['AirFieldFile'] if not vizual_options['CENCEL_AIR'] else None

        # If we do not use simple imager we probably use imager with a band so:
        wavelength_averaging = False if run_params['USE_SIMPLE_IMAGER'] else True

        # central wavelengths
        if USESWIR:
            wavelengths_micron = [vis_imager.centeral_wavelength_in_microns, swir_imager.centeral_wavelength_in_microns]
            # wavelengths_micron will hold the vis swir wavelengths. It will be convinient to use in some loops
        else:
            wavelengths_micron = [vis_imager.centeral_wavelength_in_microns]

        CloudFieldFile = run_params['CloudFieldFile'].replace('{CLOUD_INDEX}', cloud_index)
        atmosphere = CloudCT_setup.Prepare_Medium(CloudFieldFile=CloudFieldFile,
                                                  AirFieldFile=AirFieldFile,
                                                  MieTablesPath=MieTablesPath,
                                                  wavelengths_micron=wavelengths_micron,
                                                  wavelength_averaging=wavelength_averaging,
                                                  mie_options=run_params['mie_options'])

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
        fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['Rsat'])))

        vis_cnx = vis_cny = 32  # int(np.floor(L / vis_pixel_footprint))
        if USESWIR:
            swir_cnx = swir_cny = int(np.floor(L / swir_pixel_footprint))

        CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

        # Sometimes it is more convenient to use wide fov to see the whole cloud from all the view points.
        # so the FOV is also tuned:
        # -- TUNE FOV, CNY,CNX:
        if run_params['IFTUNE_CAM']:
            L *= 1.5
            fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['Rsat'])))
            vis_cnx = vis_cny = 32  # int(np.floor(L / vis_pixel_footprint))
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
            LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

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
            rigid_sampling = run_params['vis_options']['rigid_sampling']  # True of False
            vis_CloudCT_VIEWS, _ = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP,
                                                        ORBIT_ALTITUDE=run_params['Rsat'],
                                                        SAT_LOOKATS=SAT_LOOKATS,
                                                        Imager_config=vis_imager_config,
                                                        imager=vis_imager,
                                                        samples_per_pixel=vis_rays_per_pixel,
                                                        rigid_sampling=rigid_sampling,
                                                        VISSETUP=vizual_options['VISSETUP'])
            if USESWIR:
                swir_rays_per_pixel = run_params['swir_options']['rays_per_pixel']
                rigid_sampling = run_params['swir_options']['rigid_sampling']  # True of False
                swir_CloudCT_VIEWS, _ = CloudCT_setup.Create(SATS_NUMBER=SATS_NUMBER_SETUP,
                                                             ORBIT_ALTITUDE=run_params['Rsat'],
                                                             SAT_LOOKATS=SAT_LOOKATS,
                                                             Imager_config=swir_imager_config,
                                                             imager=swir_imager,
                                                             samples_per_pixel=swir_rays_per_pixel,
                                                             rigid_sampling=rigid_sampling,
                                                             VISSETUP=vizual_options['VISSETUP'])

            # Generate a solver array for a multi-spectral solution.
            # it is great that we can use the parallel solution of all solvers.
            # -----IMPORTANT NOTE---------
            # Right now the /numerical parameter of vis and swir are the same:

            rte_solvers = shdom.RteSolverArray()

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

            # -----------------------------------------------
            # ---------RENDER IMAGES FOR CLOUDCT SETUP ------
            # -----------------------------------------------
            """
            Each projection in CloudCT_VIEWS is A Perspective projection (pinhole camera).
            The method CloudCT_VIEWS.update_measurements(...) takes care of the rendering and updating the measurments.
            """

            # the order of the bands is important here.
            if USESWIR:
                CloudCT_measurements = CloudCT_setup.SpaceMultiView_Measurements(
                    [vis_CloudCT_VIEWS, swir_CloudCT_VIEWS])
            else:
                CloudCT_measurements = CloudCT_setup.SpaceMultiView_Measurements([vis_CloudCT_VIEWS])

            if not run_params['USE_SIMPLE_IMAGER']:
                # If we do not use simple imager we probably use imager such that noise can be applied:
                reduce_exposure = True
                scale_ideally = False
                apply_noise = True

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
                                                       IF_APPLY_NOISE=apply_noise)

            # Calculate irradiance of the spesific wavelength:
            # use plank function:
            Cosain = np.cos(np.deg2rad((180 - run_params['sun_zenith'])))
            temp = 5900  # K

            L_TOA_vis = 6.8e-5 * 1e-9 * plank(1e-6 * np.array(CloudCT_measurements._imagers_unique_wavelengths_list[0]),
                                              temp)  # units fo W/(m^2)
            solar_flux_vis = L_TOA_vis * Cosain
            # scale the radiance:
            vis_max_radiance_list = [image.max() * solar_flux_vis for image in
                                     CloudCT_measurements._Radiances_per_imager[0]]

            if USESWIR:
                L_TOA_swir = 6.8e-5 * 1e-9 * plank(
                    1e-6 * np.array(CloudCT_measurements._imagers_unique_wavelengths_list[1]), temp)  # units fo W/(m^2)
                solar_flux_swir = L_TOA_swir * Cosain
                # scale the radiance:
                swir_max_radiance_list = [image.max() * solar_flux_swir for image in
                                          CloudCT_measurements._Radiances_per_imager[1]]

            else:
                swir_max_radiance_list = None

            # The simulate_measurements() simulate images in gray levels.
            # Now we need to save that images and be able to load that images in the inverse pipline/

            # See the simulated images:
            if forward_options['SEE_IMAGES']:
                CloudCT_measurements.show_measurments(title_content=run_params['sun_zenith'])
                plt.show()

            # save images as mat for cloudCT neural network
            result = {'satellites_images': np.array(CloudCT_measurements.images[0])}
            filename = os.path.join(run_params['neural_network']['satellites_images_path'],
                                    f'satellites_images_{cloud_index}.mat')
            sio.savemat(filename, result)

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

            cmd = create_inverse_command(run_params=run_params, inverse_options=inverse_options,
                                         vizual_options=vizual_options,
                                         forward_dir=forward_dir, AirFieldFile=AirFieldFile,
                                         run_type=run_type, log_name=log_name)

            dump_run_params(run_params=run_params, dump_dir=forward_dir)

            write_to_run_tracker(forward_dir=forward_dir, msg=[time.strftime("%d-%b-%Y-%H:%M:%S"), log_name, cmd])

            logger.debug(f'Starting Inverse: {cmd}')
            log2load = os.path.join(forward_dir, 'logs', log_name + 'USER_CHOOSE_TIME')
            logger.debug(f'tensorboard command: tensorboard --logdir {log2load} --bind_all')

            _ = subprocess.call(cmd, shell=True)

            # Time to show the results in 3D visualization:
            if inverse_options['VIS_RESULTS3D']:
                # TODO wavelength_micron=wavelengths_micron[0] is just placeholder for?
                Final_results_3Dfiles = visualize_results(forward_dir=forward_dir, log_name=log_name,
                                                          wavelength_micron=wavelengths_micron[0])

            # save lwc and reff for neural network
            copyfile(Final_results_3Dfiles[0], os.path.join(run_params['neural_network']['lwcs_path'],
                                                            Final_results_3Dfiles[0].split('/')[-1]))
            copyfile(Final_results_3Dfiles[1], os.path.join(run_params['neural_network']['reffs_path'],
                                                            Final_results_3Dfiles[1].split('/')[-1]))

            logger.debug("Inverse phase complete")

        logger.debug(f"--------------- End for cloud {cloud_index} , {time.time()-start_time} sec---------------")

    return vis_max_radiance_list, swir_max_radiance_list


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

    times = [i.split(logs_prefix + '-')[-1].replace('_gradients', '') for i in logs_files]
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

    return Final_results_3Dfiles


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
        imager_config[index] = [True]

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
                           forward_dir, AirFieldFile, run_type, log_name):
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
    INIT_USE = ' --init ' + inverse_options['init']

    GT_USE = ''
    GT_USE = GT_USE + ' --cloudct_use' if inverse_options['cloudct_use'] else GT_USE  # -----------------
    GT_USE = GT_USE + ' --add_rayleigh' if inverse_options['add_rayleigh'] and not vizual_options[
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

    OTHER_PARAMS = OTHER_PARAMS + ' --air_path ' + AirFieldFile if not vizual_options['CENCEL_AIR'] else OTHER_PARAMS

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

        # First of all we don't neet the CloudCT measurments type here sinse the extinction optimization does not
        # support inverse with more than one wavelength.
        GT_USE = GT_USE.replace('--cloudct_use', '')  # it will erase this flag either it is on or off.
        # Convert the CloudCT_measurements to regular measurements and save it: 
        medium, solver, CloudCT_measurements = shdom.load_CloudCT_measurments_and_forward_model(forward_dir)
        measurements = CloudCT_measurements.convert2regular_measurements()
        shdom.save_forward_model(forward_dir, medium, solver, measurements)
        # --------------------------------------------------------------------------

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


if __name__ == '__main__':
    clouds_path = "/home/yaelsc/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"
    num_workers = 10
    cloud_indices_chunks = np.array_split([i.split('clouds/cloud')[1].split('.txt')[0] for i in glob.glob(clouds_path)],num_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_url = {executor.submit(main, cloud_indices_chunks[i]) for i in np.arange(num_workers)}
