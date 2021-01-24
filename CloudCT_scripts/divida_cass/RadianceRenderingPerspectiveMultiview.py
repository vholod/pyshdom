import glob
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

import shdom
from CloudCT_scripts.CloudCT_simulate_for_nn import load_run_params

# Mie scattering for water droplets
from shdom import SpaceMultiView

mie = shdom.MiePolydisperse()
mie.read_table(
    file_path='/home/yaelsc/PycharmProjects/pyshdom/yael/mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
index = 0
data_dir = '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/clouds_divided_10'
for file_name in glob.iglob(
        '/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/cloud*.txt'):
    start_process_time = time.time()
    cloud_index = file_name.split('/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/cloud')[1].split('.txt')[0]

    print(f'start processing cloud {cloud_index}')

    droplets.load_from_csv_divided(file_name, veff=0.1)
    # threshold
    run_params = load_run_params(params_path="run_params_cloud_ct_nn_rico.yaml")
    mie_options = run_params['mie_options']
    droplets.reff.data[droplets.reff.data <= mie_options['start_reff']] = mie_options['start_reff']
    droplets.reff.data[droplets.reff.data >= mie_options['end_reff']] = mie_options['end_reff']
    if len(droplets.veff.data[droplets.veff.data <= mie_options['start_veff']]) > 0:
        droplets.veff.data[droplets.veff.data <= mie_options['start_veff']] = mie_options['start_veff']
    if len(droplets.veff.data[droplets.veff.data >= mie_options['end_veff']]) > 0:
        droplets.veff.data[droplets.veff.data >= mie_options['end_veff']] = mie_options['end_veff']

    droplets.add_mie(mie)

    # extract the extinction:
    extinction = mie.get_extinction(droplets.lwc, droplets.reff, droplets.veff)
    extinction_data = extinction.data  # 1/km

    # save extintcion as mat file:
    sio.savemat(os.path.join(data_dir,"lwcs",f"cloud{cloud_index}.mat"),
                dict(beta=extinction_data, lwc=droplets.lwc.data, reff=droplets.reff.data, veff=droplets.veff.data))
    print("saving the lwc .mat file to: {}".format(os.path.join(data_dir,"lwcs",f"cloud{cloud_index}.mat")))

    # Rayleigh scattering for air molecules up to 20 km
    df = pd.read_csv('../../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
    altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
    temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
    temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
    air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
    rayleigh = shdom.Rayleigh(wavelength=0.672)
    rayleigh.set_profile(temperature_profile.resample(air_grid))
    air = rayleigh.get_scatterer()

    atmospheric_grid = droplets.grid + air.grid
    atmosphere = shdom.Medium(atmospheric_grid)
    atmosphere.add_scatterer(droplets, name='cloud')
    atmosphere.add_scatterer(air, name='air')

    numerical_params = shdom.NumericalParameters(num_mu_bins=16, num_phi_bins=32, adapt_grid_factor=5,
                                                 split_accuracy=0.1,
                                                 max_total_mb=300000.0, num_sh_term_factor=5)
    scene_params = shdom.SceneParameters(wavelength=mie.wavelength, source=shdom.SolarSource(azimuth=180, zenith=157.5))

    rte_solver = shdom.RteSolver(scene_params, numerical_params)
    rte_solver.set_medium(atmosphere)

    rte_solver.solve(maxiter=2)

    dx = atmospheric_grid.dx
    dy = atmospheric_grid.dy

    nz = atmospheric_grid.nz
    nx = atmospheric_grid.nx
    ny = atmospheric_grid.ny

    Lx = atmospheric_grid.bounding_box.xmax - atmospheric_grid.bounding_box.xmin
    Ly = atmospheric_grid.bounding_box.ymax - atmospheric_grid.bounding_box.ymin
    Lz = atmospheric_grid.bounding_box.zmax - atmospheric_grid.bounding_box.zmin
    L = max(Lx, Ly)
    Lz_droplets = droplets.grid.bounding_box.zmax - droplets.grid.bounding_box.zmin
    dz = Lz_droplets / nz

    # USED FOV, RESOLUTION and SAT_LOOKATS:
    PIXEL_FOOTPRINT = 0.05  # km
    Rsat = 600
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
    cny = 64 # 32  # int(np.floor(L / PIXEL_FOOTPRINT))
    cnx = 64 # 32  # int(np.floor(L / PIXEL_FOOTPRINT))

    CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]

    sc = 50  # 50^2 rays from each pixel
    ny = sc * cny
    nx = sc * cnx

    origin = [[0.5 * cnx * dx, 0.5 * cny * dy, Rsat],
              [0.5 * cnx * dx, 0.5 * cny * dy, Rsat]]

    lookat = [[0.5 * cnx * dx, 0.5 * cny * dy, 0],
              [0.5 * cnx * dx, 0.5 * cny * dy, 0]]
    SATS_NUMBER = 10
    SAT_LOOKATS = []
    for _ in np.arange(SATS_NUMBER):
        SAT_LOOKATS.append([0.5 * cnx * dx, 0.5 * cny * dy, 0])

    projection = SpaceMultiView.Create(SATS_NUMBER=SATS_NUMBER, ORBIT_ALTITUDE=Rsat, CAM_FOV=fov, CAM_RES=(cnx, cny),
                                       SAT_LOOKATS=SAT_LOOKATS, VISSETUP=False)

    camera = shdom.Camera(shdom.RadianceSensor(), projection)
    images = camera.render(rte_solver, n_jobs=40)

    result = {'satellites_images': images,
              'file_name': file_name,
              'sat_lookats': SAT_LOOKATS}
    filename = os.path.join(data_dir,"satellites_images",f"satellites_images_{cloud_index}.mat")
    sio.savemat(filename, result)

    plt.figure()
    f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
    for ax, image in zip(axarr, images):
        ax.imshow(image, cmap='gray')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.axis('off')
    plt.show()

    print(f'finished process of cloud {cloud_index}, time was: {(time.time() - start_process_time)} seconds')
