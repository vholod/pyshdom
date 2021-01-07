import dill as pickle
import scipy.io as sio
import shdom
import scipy.ndimage as ndimage
import numpy as np

measurements_path = "/home/yaelsc/PycharmProjects/pyshdom/CloudCT_scripts/cloud_fields/measurments.pkl"
grid_path = "/home/yaelsc/PycharmProjects/pyshdom/CloudCT_scripts/cloud_fields/grid.pkl"

with open(measurements_path,'rb') as f:
    measurements = pickle.load(f)

with open(grid_path,'rb') as f:
    grid = pickle.load(f)

satellites_images_path = "/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/satellites_images/satellites_images_89.mat"
satellites_images = sio.loadmat(satellites_images_path)["satellites_images"]
measurements._images = satellites_images

radiance_threshold = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
carver = shdom.SpaceCarver(measurements)
mask = carver.carve(grid, agreement=0.7, thresholds=radiance_threshold)
mask._data = (ndimage.morphology.binary_closing(mask.data, np.ones((3, 3, 3)))).astype(bool)


print('bla')