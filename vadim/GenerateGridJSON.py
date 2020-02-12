import os 
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import json

from collections import OrderedDict
data=OrderedDict()
""" 
A Grid object defining the 3D or 1D grid of the atmopshere. 

A 3D Grid can be defined with:
  1. x, y, z grids.
  2. A BoundingBox and grid resolution (nx, ny, nz).

A 1D grid is defined with a z grid.

Parameters
----------
bounding_box: BoundingBox, optional
    A BoundingBox object for 3D grid. If specified nx, ny and nz must be specified as well.
nx: integer, optional
    Number of grid point in x axis. Must be specified with a BoundingBox object.
ny: integer, optional
    Number of grid point in y axis. Must be specified with a BoundingBox object.
nz: integer, optional
    Number of grid point in z axis. Must be specified with a BoundingBox object.
x: np.array(dtype=float, shape=(nx,)), optional
    Grid along x axis. Must be specified with y,z grids.
y: np.array(dtype=float, shape=(ny,)), optional
    Grid along y axis. Must be specified with x,z grids. 
z: np.array(dtype=float, shape=(nz,)), optional
    Grid along z axis. Either specified with x,y grids (3D grid) or by itself (1D grid).
"""
exif_data_file = 'dannys_clouds/GRID_SMALL_MED_view55.json'
data['Grid_type'] = "3D"
Grid_bounding_box=OrderedDict()
Grid_bounding_box['xmin'] = -1460*1e-3 # must be in km
Grid_bounding_box['xmax'] = (780)*1e-3
Grid_bounding_box['ymin'] = -680*1e-3
Grid_bounding_box['ymax'] = (2580)*1e-3
Grid_bounding_box['zmin'] = 0*1e-3
Grid_bounding_box['zmax'] = (4000-20)*1e-3
Grid_bounding_box['nz'] = 200
Grid_bounding_box['nx'] = 113
Grid_bounding_box['ny'] = 164
Grid_bounding_box['dz'] = 20*1e-3 # must be in km
Grid_bounding_box['dx'] = 20*1e-3 # must be in km
Grid_bounding_box['dy'] = 20*1e-3 # must be in km
data['Grid_bounding_box'] = Grid_bounding_box
with open(exif_data_file, 'w') as f:
    json.dump(data, f, indent=4) 