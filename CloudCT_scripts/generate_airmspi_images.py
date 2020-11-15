import numpy as np
import pandas as pd
import scipy.io as sio
import shdom
import glob
import time
import matplotlib.pyplot as plt

from shdom.AirMSPI import AirMSPIMeasurements

mie = shdom.MiePolydisperse()
mie.read_table(file_path="/home/yaelsc/PycharmProjects/pyshdom/mie_tables/polydisperse/Water_660nm.scat")

droplets = shdom.MicrophysicalScatterer()
for file_name in glob.glob("/home/yaelsc/Data/AirMSPI/clouds_airmspi/cloud*.txt"):
    start_process_time = time.time()
    cloud_index = file_name.split("/home/yaelsc/Data/AirMSPI/clouds_airmspi/cloud")[-1].split(".txt")[0]
    try:
        droplets.load_from_csv(file_name,veff=0.1)

        lwc = droplets.lwc.data
        veff = droplets.veff.data
        reff = droplets.reff.data

        new_lwc = shdom.GridData(droplets.grid,lwc/25)
        new_veff = shdom.GridData(droplets.grid,veff)
        new_reff = shdom.GridData(droplets.grid,reff)

        new_droplets = shdom.MicrophysicalScatterer(new_lwc,new_reff,new_veff)
        droplets = new_droplets
        droplets.add_mie(mie)

        # Rayleigh scattering for air molecules up to 20 km
        df = pd.read_csv("/home/yaelsc/PycharmProjects/pyshdom/ancillary_data/AFGL_summer_mid_lat.txt", comment='#', sep=' ')
        altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
        temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
        temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
        air_grid = shdom.Grid(z=np.linspace(0, 20, 20))
        rayleigh = shdom.Rayleigh(wavelength=0.660)
        rayleigh.set_profile(temperature_profile.resample(air_grid))
        air = rayleigh.get_scatterer()

        atmospheric_grid = droplets.grid + air.grid
        atmosphere = shdom.Medium(atmospheric_grid)
        atmosphere.add_scatterer(droplets, name='cloud')
        atmosphere.add_scatterer(air, name='air')

        numerical_params = shdom.NumericalParameters(num_mu_bins=8, num_phi_bins=16, adapt_grid_factor=5,
                                                     split_accuracy=0.1,
                                                     max_total_mb=300000.0, num_sh_term_factor=5)
        scene_params = shdom.SceneParameters(
            surface=shdom.LambertianSurface(albedo=0.05),
            wavelength=mie.wavelength,
            source=shdom.SolarSource(azimuth=0, zenith=132.7))

        rte_solver = shdom.RteSolver(scene_params, numerical_params)
        rte_solver.set_medium(atmosphere)

        # rte_solver.init_solution()
        rte_solver.solve(maxiter=150)

        zenith_list = [70.5, 60, 45.6, 26.1, 0, 26.1, 45.6, 60, 70.5]
        azimuth_list = [90, 90, 90, 90, 0, -90, -90, -90, -90]

        M = AirMSPIMeasurements()
        rois = [[1288, 1608, 588, 816], [1356, 1680, 640, 892], [1416, 1756, 696, 940], [1492, 1820, 732, 992],
                [1540, 1884, 792, 1036], [1604, 1944, 828, 1076], [1660, 2012, 884, 1120], [1728, 2084, 936, 1172],
                [1792, 2152, 980, 1224]]
        valid_wavelength = [660]
        data_dir = '/home/yaelsc/Data/AirMSPI/raw_data'
        M.load_from_hdf(data_dir,region_of_interest=rois,valid_wavelength=valid_wavelength)
        camera = M.camera
        images = camera.render(rte_solver, n_jobs=40)

        result = {'satellites_images': images,
                  'file_name': file_name}
        filename = f'/home/yaelsc/Data/AirMSPI/satellites_images/satellites_images_{cloud_index}.mat'
        sio.savemat(filename, result)

        plt.figure()
        f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
        for ax, image in zip(axarr, images):
            ax.imshow(image, cmap='gray')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.axis('off')
        plt.show()

        print(f"{cloud_index} finished in {time.time()-start_process_time}")
    except Exception as e:
        print(f"cloud {cloud_index} failed : {e}")
