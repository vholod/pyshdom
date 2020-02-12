import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
import pickle
import mayavi.mlab as mlab
import glob

# Mie scattering for water droplets
mie = shdom.MiePolydisperse()
mie.read_table(file_path='../../mie_tables/polydisperse/Water_672nm.scat')

# Generate a Microphysical medium
droplets = shdom.MicrophysicalScatterer()
droplets.load_from_csv('../../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
droplets.add_mie(mie)

Grid_bounding_box = droplets.bounding_box
dx = droplets.lwc.grid.dx
dy = droplets.lwc.grid.dy

Grid_shape = droplets.grid.shape
xgrid = np.linspace(Grid_bounding_box.xmin, Grid_bounding_box.xmax,Grid_shape[0])
ygrid = np.linspace(Grid_bounding_box.ymin, Grid_bounding_box.ymax,Grid_shape[1])
zgrid = np.linspace(Grid_bounding_box.zmin, Grid_bounding_box.zmax,Grid_shape[2])   

X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
LWC_MAT = droplets.lwc.data
RE_MAT = droplets.reff.data

from scipy.ndimage import zoom
# padding to fit 40x40x40
npad = ((4, 4), (1, 2), (7, 7))
LWC_MAT_padded = np.pad(LWC_MAT, pad_width=npad, mode='constant', constant_values=0.0)
npad = ((7, 7))# reff is 1D array, varies only with height
RE_MAT_padded = np.pad(RE_MAT, pad_width=npad, mode='constant', constant_values=0.0)
savetxt = True
if(1):
    # change resolution
    #zoom_ms = np.array([0.25, 0.5, 1, 2, 4])
    zoom_ms = np.array([ (15/40), (30/40), (50/40), (100/40)])
    LWC_MATS = []
    RE_MATS = []
    for zoom_option in zoom_ms:
        # zoom the medium:
        tmpvol_lwc = zoom(LWC_MAT_padded, zoom_option, mode='nearest')
        tmpvol_lwc[tmpvol_lwc < 1e-4] = 0
        
        LWC_MATS.append(tmpvol_lwc)
        tmpvol_re = zoom(RE_MAT_padded, zoom_option, mode='nearest')
        tmpvol_re[tmpvol_re < 0.05] = 0
        
        RE_MATS.append(tmpvol_re)
        # set cloud bottom to zero:
        zgrid = np.linspace(0, Grid_bounding_box.zmax - Grid_bounding_box.zmin,tmpvol_lwc.shape[2])   
        
        print("current lwc size is {}".format(tmpvol_lwc.shape))
        #print("current re size is {}".format(tmpvol_re.shape))
        
        if(savetxt):
            file_name = "tamar_cvpr_rico{}x{}x{}.txt"\
                .format(tmpvol_lwc.shape[0],tmpvol_lwc.shape[1],tmpvol_lwc.shape[2])
            # create the txt files:
            comment_line = "tamar cvpr"
            np.savetxt(file_name, X=np.array([tmpvol_lwc.shape]), fmt='%d', header=comment_line)
            f = open(file_name, 'ab') 
            np.savetxt(f, X=np.concatenate((np.array([(1/zoom_option)*dx, (1/zoom_option)*dy]), zgrid)).reshape(1,-1), fmt='%2.3f')
            nx, ny, nz = tmpvol_lwc.shape
            lwc = tmpvol_lwc
            
            reff = np.tile(tmpvol_re[np.newaxis, np.newaxis, :], (nx, ny, 1))
            y, x, z = np.meshgrid(range(ny), range(nx), range(nz))
            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), lwc.ravel(), reff.ravel())).T
            np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f')        
            f.close()    
        
    

"""
overall beta sizes should be:
10x10x10
20x20x20
40x40x40
80x80x80
160x160x160
"""

# Test what I got:
txt_files = sorted(glob.glob('./tamar_cvpr_rico*.txt'))
txt_files_names = [os.path.split(i)[-1] for i in txt_files]
for file in txt_files_names:
    mlab.figure(size=(600, 600))
    droplets = shdom.MicrophysicalScatterer()
    droplets.load_from_csv(file, veff=0.1)
    droplets.add_mie(mie)
    
    Grid_bounding_box = droplets.bounding_box
    Grid_shape = droplets.grid.shape
    xgrid = np.linspace(Grid_bounding_box.xmin, Grid_bounding_box.xmax,Grid_shape[0])
    ygrid = np.linspace(Grid_bounding_box.ymin, Grid_bounding_box.ymax,Grid_shape[1])
    zgrid = np.linspace(Grid_bounding_box.zmin, Grid_bounding_box.zmax,Grid_shape[2])     

    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    LWC_MAT = droplets.lwc.data
    RE_MAT = droplets.reff.data  
    print("-- current lwc size is {}".format(LWC_MAT.shape))
    nx, ny, nz = LWC_MAT.shape
    
    dx = droplets.lwc.grid.dx
    dy = droplets.lwc.grid.dy
    dz = (Grid_bounding_box.zmax- Grid_bounding_box.zmin)/(nz-1)
    print("-- current dx,dy,dz are {},{},{}".format(dx,dy,dz))
    
    scf = mlab.pipeline.scalar_field(X, Y, Z,LWC_MAT)
    scf.spacing = [dx, dy, dz]
    scf.update_image_data = True 
        
    figh = mlab.gcf()
    #mlab.pipeline.volume(scf, figure=figh) # no working on servers
    isosurface = mlab.pipeline.iso_surface(scf, contours=[0.0001],color = (1, 1,1),opacity=0.1,transparent=True)
    mlab.pipeline.volume(isosurface, figure=figh)
    mlab.pipeline.image_plane_widget(scf,plane_orientation='x_axes',slice_index=10)
                                
    figh.scene.anti_aliasing_frames = 0
    mlab.scalarbar()
    mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 
    
    mlab.show() 