import dill as pickle
import scipy.io as sio
import shdom
import scipy.ndimage as ndimage
import numpy as np
import glob
import time


measurements_path = "/home/yaelsc/PycharmProjects/pyshdom/CloudCT_scripts/cloud_fields/measurments.pkl"
grid_path = "/home/yaelsc/PycharmProjects/pyshdom/CloudCT_scripts/cloud_fields/grid.pkl"

with open(measurements_path,'rb') as f:
    measurements = pickle.load(f)

with open(grid_path,'rb') as f:
    grid = pickle.load(f)

radiance_threshold = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

satellites_images_paths = "/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/satellites_images/satellites_images_*.mat"
for satellites_images_path in glob.glob(satellites_images_paths):
    cloud_index = satellites_images_path.split('/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/satellites_images/satellites_images_')[1].split('.mat')[0]
    print(f"start cloud {cloud_index}")

    satellites_images = sio.loadmat(satellites_images_path)["satellites_images"]
    measurements._images = satellites_images

    carver = shdom.SpaceCarver(measurements)

    start_time = time.time()
    mask = carver.carve(grid, agreement=0.7, thresholds=radiance_threshold)
    mask._data = (ndimage.morphology.binary_closing(mask.data, np.ones((3, 3, 3)))).astype(bool)

    mask_path = f"/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/masks/mask_{cloud_index}.mat"
    sio.savemat(mask_path, dict(mask=mask.data, time=(time.time()-start_time)))

    print(f"saved mask of cloud {cloud_index}, process time of {(time.time()-start_time)} seconds")

