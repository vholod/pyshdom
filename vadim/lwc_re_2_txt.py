import os 
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
from argparse import ArgumentParser
import json

TFLAG = False
def exists(p, msg):
    assert os.path.exists(p), msg
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--lwc', type=str,
                        dest='lwc_path',
                        help='path to the lwc mat file',
                        metavar='lwc_path', required=True)
    
    parser.add_argument('--re', type=str,
                        dest='re_path',
                        help='path to the re (effective radius) mat file',
                        metavar='re_path', required=True)
    
    parser.add_argument('--c', type=str,
                        dest='if_convert_kg2g',
                        help='if c=1, convert the LWC from kg/m^3 to g/m^3',
                        metavar='if_convert_kg2g', default=False)    
    
    parser.add_argument('--grid_data', type=str,
                        dest='json_grid_file',
                        help='path to the json file which explaines the grids of the mat files',
                        metavar='json_grid_file', required=True)    
    
    parser.add_argument('--ve', type=str,
                        dest='ve_path',
                        help='path to the ve (effective variance) mat file',
                        metavar='ve_path', default=None)    
    
    parser.add_argument('--output', type=str,
                        dest='out_path',
                        help='path csv (txt) file',
                        metavar='out_path', default="./mediume_data.txt")   
    
    opts = parser.parse_args()
    
    exists(opts.lwc_path, 'lwc_path not found!')
    print("loading the 3D mat from: {}".format(opts.lwc_path))
    lwc = sio.loadmat(opts.lwc_path)['lwc'] 
    if(opts.if_convert_kg2g):
        lwc = lwc*1e-3
        
    if TFLAG:
        lwc = np.transpose(lwc, (1, 0, 2))# Vadim created the mat files in matlab, thus it is important to reshape the matrix and change x,y of matlab mat.
    exists(opts.re_path, 're_path not found!')
    print("loading the 3D mat from: {}".format(opts.re_path))
    reff = sio.loadmat(opts.re_path)['re'] 
    if TFLAG:
        reff = np.transpose(reff, (1, 0, 2))# Vadim created the mat files in matlab, thus it is important to reshape the matrix and change x,y of matlab mat.
    if(not (opts.ve_path is None)):
        exists(opts.ve_path, 've_path not found!')
        print("loading the 3D mat from: {}".format(opts.ve_path))
        veff = sio.loadmat(opts.ve_path)['ve']     
        if TFLAG:
            veff = np.transpose(veff, (1, 0, 2))# Vadim created the mat files in matlab, thus it is important to reshape the matrix and change x,y of matlab mat.
    else:
        veff=0.1 # this is the defult,
     
    comment_line = "lwc from {}, re from {}, ve from {}".format(opts.lwc_path,opts.re_path,opts.ve_path)   
    
    # extrarc grid data:
    json_grid_file = opts.json_grid_file
    with open(json_grid_file, 'r') as f:
        grid_json = json.load(f) 
        
    if TFLAG:
        bounding_box = shdom.BoundingBox(grid_json['Grid_bounding_box']['xmin'],
                                     grid_json['Grid_bounding_box']['ymin'],
                                     grid_json['Grid_bounding_box']['zmin'],
                                     grid_json['Grid_bounding_box']['xmax'],
                                     grid_json['Grid_bounding_box']['ymax'],
                                     grid_json['Grid_bounding_box']['zmax'])

        nx=grid_json['Grid_bounding_box']['nx']
        ny=grid_json['Grid_bounding_box']['ny']
        nz=grid_json['Grid_bounding_box']['nz']
    else:
        bounding_box = shdom.BoundingBox(grid_json['Grid_bounding_box']['ymin'],
                                     grid_json['Grid_bounding_box']['xmin'],
                                     grid_json['Grid_bounding_box']['zmin'],
                                     grid_json['Grid_bounding_box']['ymax'],
                                     grid_json['Grid_bounding_box']['xmax'],
                                     grid_json['Grid_bounding_box']['zmax'])

        nx=grid_json['Grid_bounding_box']['ny']
        ny=grid_json['Grid_bounding_box']['nx']
        nz=grid_json['Grid_bounding_box']['nz']
        
        # calculate extinction:
        #grid = shdom.Grid(bounding_box=bounding_box,nx=nx,
                          #ny=ny,
                          #nz=nz)   
        #veff = veff * np.ones_like(reff)
        #lwc=shdom.GridData(grid, lwc).squeeze_dims()
        #reff=shdom.GridData(grid, reff).squeeze_dims() 
        #veff=shdom.GridData(grid, veff).squeeze_dims() 
        
        #mie = shdom.MiePolydisperse()
        #extinction = mie.get_extinction(lwc, reff, veff)         
        
        
    zgrid = np.linspace(grid_json['Grid_bounding_box']['zmin'], grid_json['Grid_bounding_box']['zmax'],nz)
    
    dx = grid_json['Grid_bounding_box']['dx'] 
    dy = grid_json['Grid_bounding_box']['dy'] 
    dz = grid_json['Grid_bounding_box']['dz']
    if TFLAG:
        assert dx == (grid_json['Grid_bounding_box']['xmax']- grid_json['Grid_bounding_box']['xmin'])/(nx-1), \
               'dab data discription'
        assert dy == (grid_json['Grid_bounding_box']['ymax']- grid_json['Grid_bounding_box']['ymin'])/(ny-1), \
               'dab data discription'  
        assert dz == (grid_json['Grid_bounding_box']['zmax']- grid_json['Grid_bounding_box']['zmin'])/(nz-1), \
               'dab data discription'    
    else:
        assert dy == (grid_json['Grid_bounding_box']['ymax']- grid_json['Grid_bounding_box']['ymin'])/(nx-1), \
               'dab data discription'
        assert dx == (grid_json['Grid_bounding_box']['xmax']- grid_json['Grid_bounding_box']['xmin'])/(ny-1), \
               'dab data discription'  
        assert dz == (grid_json['Grid_bounding_box']['zmax']- grid_json['Grid_bounding_box']['zmin'])/(nz-1), \
               'dab data discription'            


    #save the txt file:
    np.savetxt(opts.out_path, X=np.array([lwc.shape]), fmt='%d', header=comment_line)
    f = open(opts.out_path, 'ab') 
    np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1,-1), fmt='%2.3f')
    y, x, z = np.meshgrid(range(ny), range(nx), range(nz))
    data = np.vstack((x.ravel(), y.ravel(), z.ravel(), lwc.ravel(), reff.ravel())).T
    np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f')        
    f.close()
    
    # save the extinction:
    #out_mat = 'beta'+'_sample.mat'
    #sio.savemat(out_mat, dict(data_beta=extinction))  
        
    

    
def test_data_load():
    out_path = "/home/vhold/pyshdom/vadim/test_csv.txt"
    ref_dir = "/home/vhold/pyshdom/vadim/dannys_clouds/"
    lwc_path = os.path.join(ref_dir,"LWC_SMALL_MED_view55.mat" ) 
    re_path = os.path.join(ref_dir,"Re_SMALL_MED_view55.mat" )
    json_grid_file  = os.path.join(ref_dir,"GRID_SMALL_MED_view55.json" )
    with open(json_grid_file, 'r') as f:
        grid_json = json.load(f)    

    # extrarc grid data:
    bounding_box = shdom.BoundingBox(grid_json['Grid_bounding_box']['xmin'],
                                     grid_json['Grid_bounding_box']['ymin'],
                                     grid_json['Grid_bounding_box']['zmin'],
                                     grid_json['Grid_bounding_box']['xmax'],
                                     grid_json['Grid_bounding_box']['ymax'],
                                     grid_json['Grid_bounding_box']['zmax'])
                                     
    veff=0.1 
    lwc = sio.loadmat(lwc_path)['lwc'] 
    reff = sio.loadmat(re_path)['re'] 
    lwc = np.transpose(lwc, (1, 0, 2))# Vadim created the mat files in matlab, thus it is important to reshape the matrix and change x,y of matlab mat.
    reff = np.transpose(reff, (1, 0, 2))
    
    nx=grid_json['Grid_bounding_box']['nx']
    ny=grid_json['Grid_bounding_box']['ny']
    nz=grid_json['Grid_bounding_box']['nz']
    xgrid = np.linspace(grid_json['Grid_bounding_box']['xmin'], grid_json['Grid_bounding_box']['xmax'],nx)
    ygrid = np.linspace(grid_json['Grid_bounding_box']['ymin'], grid_json['Grid_bounding_box']['ymax'],ny)
    zgrid = np.linspace(grid_json['Grid_bounding_box']['zmin'], grid_json['Grid_bounding_box']['zmax'],nz)   
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    dx = grid_json['Grid_bounding_box']['dx'] 
    dy = grid_json['Grid_bounding_box']['dy'] 
    dz = grid_json['Grid_bounding_box']['dz']
    assert dx == (grid_json['Grid_bounding_box']['xmax']- grid_json['Grid_bounding_box']['xmin'])/(nx-1), \
           'dab data discription'
    assert dy == (grid_json['Grid_bounding_box']['ymax']- grid_json['Grid_bounding_box']['ymin'])/(ny-1), \
           'dab data discription'  
    assert dz == (grid_json['Grid_bounding_box']['zmax']- grid_json['Grid_bounding_box']['zmin'])/(nz-1), \
           'dab data discription'     
    #scf = mlab.pipeline.scalar_field(X, Y, Z,lwc)
    #figh = mlab.gcf()
    #isosurface = mlab.pipeline.iso_surface(scf, contours=[0.0001],color = (1, 1,1),opacity=0.1,transparent=True)
    #mlab.pipeline.volume(isosurface, figure=figh)
    #mlab.pipeline.image_plane_widget(scf,plane_orientation='x_axes',slice_index=10)
                                
    #figh.scene.anti_aliasing_frames = 0
    #mlab.scalarbar()
    #mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 

    #mlab.show()  
    
    
    
    
    
    grid = shdom.Grid(bounding_box=bounding_box,nx=grid_json['Grid_bounding_box']['nx'],
                      ny=grid_json['Grid_bounding_box']['ny'],
                      nz=grid_json['Grid_bounding_box']['nz'])
    
    comment_line = "just test"
    # Generate a Microphysical medium
     
    np.savetxt(out_path, X=np.array([lwc.shape]), fmt='%d', header=comment_line)
    f = open(out_path, 'ab') 
    np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1,-1), fmt='%2.3f')
    y, x, z = np.meshgrid(range(ny), range(nx), range(nz))
    data = np.vstack((x.ravel(), y.ravel(), z.ravel(), lwc.ravel(), reff.ravel())).T
    np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f')        
    f.close()
    
    # -- the following doesn't work:
    #lwc=shdom.GridData(grid, lwc).squeeze_dims()
    #reff=shdom.GridData(grid, reff).squeeze_dims() 
    #droplets = shdom.MicrophysicalScatterer(lwc, reff) 
    ##droplets.set_microphysics(lwc, reff)
    #droplets.save_to_csv( out_path, comment_line)
    
    #test visualy:
    droplets = shdom.MicrophysicalScatterer()
    droplets.load_from_csv(out_path, veff=0.1)
    Grid_bounding_box = droplets.bounding_box
    Grid_shape = droplets.grid.shape
    xgrid = np.linspace(Grid_bounding_box.xmin, Grid_bounding_box.xmax,Grid_shape[0])
    ygrid = np.linspace(Grid_bounding_box.ymin, Grid_bounding_box.ymax,Grid_shape[1])
    zgrid = np.linspace(Grid_bounding_box.zmin, Grid_bounding_box.zmax,Grid_shape[2])   
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    LWC_MAT = droplets.lwc.data
    RE_MAT = droplets.reff.data
    
    scf = mlab.pipeline.scalar_field(X, Y, Z,RE_MAT)
    figh = mlab.gcf()
    isosurface = mlab.pipeline.iso_surface(scf, contours=[0.0001],color = (1, 1,1),opacity=0.1,transparent=True)
    mlab.pipeline.volume(isosurface, figure=figh)
    mlab.pipeline.image_plane_widget(scf,plane_orientation='x_axes',slice_index=10)
                                
    figh.scene.anti_aliasing_frames = 0
    mlab.scalarbar()
    mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)") 

    mlab.show()    
    
if __name__ == '__main__':
    main()
    #test_data_load()
    
    