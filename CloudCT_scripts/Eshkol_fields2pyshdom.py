import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import shdom
from shdom import float_round
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.io as sio

import shdom
from shdom import float_round

try:
    from roipoly import RoiPoly
    # I take it from: https://github.com/jdoepfert/roipoly.py
    # Based on a code snippet originally posted by Daniel Kornhauser
    # (http://matplotlib.1069221.n5.nabble.com/How-to-draw-a-region-of-interest-td4972.html).
except:
    raise  # so do pip install roipoly

data_path = '../data/BOMEX_512x512x170_500CCN_20m_micro_256_0000021600_ONLY_RE_VE_LWC.mat'  # '../WIZ_Clouds/BOMEX_256x256x100_5000CCN_50m_micro_256_0000007740_ONLY_RE_VE_LWC_NC.mat'


# -----functions------------------------------------:
def calc_high_map(volumefield, zgrid):
    """
    Extracts top of the clouds.

    """
    nx, ny, nz = volumefield.shape
    K = np.zeros([nx, ny])
    K = np.cumsum(volumefield, axis=2)
    K = np.argmax(K, axis=2)
    high_map = zgrid[K]
    return high_map


# ask the user to set the name of the cloud field
file_name = input("Enter file name of the new cloud field: ")
file_name = file_name + '.txt'

# load 3d data:
print(f'load {data_path}')
data3d = sio.loadmat(data_path)
lwc = data3d['lwc']
reff = data3d['reff']
veff = data3d['veff']
# get relevant params
xgrid, ygrid, zgrid = np.round(data3d['x'].flatten() * 1e-3, 3), np.round(data3d['y'].flatten() * 1e-3, 3), np.round(
    data3d['z'].flatten() * 1e-3, 3)
nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
print(f"nx:{nx}, ny:{ny}, nz:{nz}")
x_min, y_min, z_min = xgrid.min(), ygrid.min(), zgrid.min()
x_max, y_max, z_max = xgrid.max(), ygrid.max(), zgrid.max()
print("x_min, y_min, z_min, x_max, y_max, z_max:", x_min, y_min, z_min, x_max, y_max, z_max)
# Compute dx, dy
dx = np.unique(np.round(np.diff(xgrid),3))
dy = np.unique(np.round(np.diff(ygrid),3))
print(f'dx:{dx},dy:{dy}')
assert (len(dx) == 1) and (len(dy) == 1), 'dx or dy are not uniform'
dx, dy = dx[0], dy[0]

# assert float_round(dx) == float_round(data3d['dx'][0][0]), 'dab data discription!'
# assert float_round(dy) == float_round(data3d['dy'][0][0]), 'dab data discription!'
# assert float_round(dz) == float_round(data3d['dz'][0][0]), 'dab data discription!'

# set grid:
bounding_box = shdom.BoundingBox(x_min, y_min, z_min, x_max, y_max, z_max)
grid = shdom.Grid(bounding_box=bounding_box, nx=nx, ny=ny, nz=nz)

# set scatterer:
lwc = shdom.GridData(grid, lwc)
reff = shdom.GridData(grid, reff)
veff = shdom.GridData(grid, veff)
droplets = shdom.MicrophysicalScatterer(lwc, reff, veff)

# set medium:
medium = shdom.Medium(grid)

# exclude data discription buges:
assert float_round(y_max) == float_round(medium.grid.ymax), 'dab data discription!'
assert float_round(x_max) == float_round(medium.grid.xmax), 'dab data discription!'
assert float_round(z_max) == float_round(medium.grid.zmax), 'dab data discription!'

assert float_round(y_min) == float_round(medium.grid.ymin), 'dab data discription!'
assert float_round(x_min) == float_round(medium.grid.xmin), 'dab data discription!'
assert float_round(z_min) == float_round(medium.grid.zmin), 'dab data discription!'

assert float_round(dx) == float_round(medium.grid.dx), 'dab data discription!'
assert float_round(dy) == float_round(medium.grid.dy), 'dab data discription!'

# add a scatterer to the medium:
medium.add_scatterer(droplets, name='clouds')
# show the medium to be sure the it was loaded correctly:
mlab.figure(size=(600, 600))
medium.show_scatterer(name='clouds')
mlab.title('full volume')

# extract lwc from medium. It will be used as the volume field from which we cut a 3d part.
volumefield = droplets.lwc.data
# calc_high_map
High_map = calc_high_map(volumefield, zgrid)

# -------------------------------------------
"""
Cut one piece from the cloud/cloud field.

1. Extract the high map of the cloud field
2. The user interactively draw a polygon within the image by clicking with the left mouse button 
to select the vertices of the polygon. To close the polygon, click with the right mouse button.
After finishing the ROI, the current figure is closed so that the execution of the code can continue.
3. The function get_mask(image) creates a binary mask for a certain ROI instance, that is, a 2D numpy 
array of the size of the image array, whose elements are True if they lie inside the ROI polygon, 
and False otherwise.
notes:
The new mediume will be padded with zeros on its outer boundaries.

"""

fig = plt.figure(figsize=(20, 20))

plt.imshow(High_map, vmin=0, vmax=np.amax(High_map))
plt.title('Full High map')
plt.gca().invert_yaxis()
my_roi = RoiPoly(color='r')  # draw new ROI in red color

# lets the user choose the roi.
my_roi.display_roi()
# Extracting a binary mask image.
mask = my_roi.get_mask(High_map)

XX = np.arange(0, nx)
YY = np.arange(0, ny)
YY, XX = np.meshgrid(XX, YY)

Xpoints = np.trim_zeros((XX[mask]).ravel())
Ypoints = np.trim_zeros((YY[mask]).ravel())

min_x_index = min(Xpoints)
min_y_index = min(Ypoints)

max_x_index = max(Xpoints)
max_y_index = max(Ypoints)

print("when cut from high map")
print("The min_x_index is {}\n".format(min_x_index))
print("The min_y_index is {}\n".format(min_y_index))
print("The max_x_index is {}\n".format(max_x_index))
print("The max_y_index is {}\n".format(max_y_index))

# I always have the problem with the rounding (e.g. 0.2400000000014 or 0.2399999999), so I use Utils.float_round
min_x_coordinates = float_round(xgrid[min_x_index])
min_y_coordinates = float_round(ygrid[min_y_index])

max_x_coordinates = float_round(xgrid[max_x_index] - dx)
max_y_coordinates = float_round(ygrid[max_y_index] - dy)
# why -dx and -dy?
# Becouse The Xmin is here ->|_|_|_|_|_|_|_|_|.
# The Xmax is one dx befor the last corner (|_|_|_|_|_|_|_->|_|). I made a lot of miskaes when I implemented it.

new_field = volumefield[min_x_index:max_x_index, min_y_index:max_y_index, :]
cumsum_volume_axis2 = np.cumsum(new_field, axis=2, dtype=float)
# GET RID OF ZEROS: fined the minimum excluding zeros, 
tmp = np.sum(np.sum(cumsum_volume_axis2, axis=1), axis=0)
tmp[tmp == 0] = max(tmp)
# GET RID OF ZEROS -> done
Bottom_minimum = np.argmin(tmp)

min_z_index = Bottom_minimum
max_z_index = np.amax(np.argmax(cumsum_volume_axis2, axis=2))

new_field = new_field[:, :, min_z_index:max_z_index]

# I always do the padding. I Pad with zeros on the sides.
# So:

min_z_coordinates = float_round(zgrid[min_z_index])
max_z_coordinates = float_round(zgrid[max_z_index-1]) # TODO makesure equivalent to -dz (40*1e-3) instead of -1 index

# The +- ds is becouse of the padding
min_x_coordinates = float_round(min_x_coordinates - dx)
max_x_coordinates = float_round(max_x_coordinates + dx)

min_y_coordinates = float_round(min_y_coordinates - dy)
max_y_coordinates = float_round(max_y_coordinates + dy)

Yrange = [min_x_coordinates, max_x_coordinates]
Xrange = [min_y_coordinates, max_y_coordinates]
Zrange = [min_z_coordinates, max_z_coordinates]
# The + 2 is becouse of the padding
new_nx = max_x_index - min_x_index + 2
new_ny = max_y_index - min_y_index + 2
new_nz = max_z_index - min_z_index

print("After the cut from high map")
print("The min_x_coordinates is {}\n".format(min_x_coordinates))
print("The min_y_coordinates is {}\n".format(min_y_coordinates))
print("The max_x_coordinates is {}\n".format(max_x_coordinates))
print("The max_y_coordinates is {}\n".format(max_y_coordinates))

# -------------------------------------------------------------
# set every thing new:
# set scatterer:
new_lwc = lwc.data[min_x_index:max_x_index, min_y_index:max_y_index, min_z_index:max_z_index]
new_reff = reff.data[min_x_index:max_x_index, min_y_index:max_y_index, min_z_index:max_z_index]
new_veff = veff.data[min_x_index:max_x_index, min_y_index:max_y_index, min_z_index:max_z_index]

# I always do the padding. I Pad with zeros on the sides.
# So:
new_lwc = np.pad(new_lwc, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
new_reff = np.pad(new_reff, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)
new_veff = np.pad(new_veff, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)

# set new grid:
new_bounding_box = shdom.BoundingBox(
    min_x_coordinates, min_y_coordinates, min_z_coordinates,
    max_x_coordinates, max_y_coordinates, max_z_coordinates)

new_grid = shdom.Grid(bounding_box=new_bounding_box, nx=new_nx, ny=new_ny, nz=new_nz)

new_lwc = shdom.GridData(new_grid, new_lwc)
new_reff = shdom.GridData(new_grid, new_reff)
new_veff = shdom.GridData(new_grid, new_veff)
new_droplets = shdom.MicrophysicalScatterer(new_lwc, new_reff, new_veff)
# set medium:
new_medium = shdom.Medium(new_grid)
# add a scatterer to the medium:
new_medium.add_scatterer(new_droplets, name='clouds')
# show the medium to be sure the it was loaded correctly:
mlab.figure(size=(600, 600))
new_medium.show_scatterer(name='clouds')
mlab.title('new volume')

# save the txt file:
new_droplets.save_to_csv(file_name, comment_line='a part of the data of Eshkol april 2020')
# Test the creation of new txt file:
test_droplets = shdom.MicrophysicalScatterer()
test_droplets.load_from_csv(file_name)
test_medium = shdom.Medium(test_droplets.grid)
# add a scatterer to the medium:
test_medium.add_scatterer(test_droplets, name='clouds')
# show the medium to be sure the it was loaded correctly:
mlab.figure(size=(600, 600))
test_medium.show_scatterer(name='clouds')
mlab.title('test volume')

mlab.show()

print('done')
