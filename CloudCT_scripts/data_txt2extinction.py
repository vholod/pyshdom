import os 
import mayavi.mlab as mlab
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shdom
from argparse import ArgumentParser
import json


def exists(p, msg):
    assert os.path.exists(p), msg
    
"""
save an extinction with 1/km units.
take inputs:
scat file
txt file describing the lwc and reff

"""
def main():
    parser = ArgumentParser()
    
    parser.add_argument('--scat', type=str,
                        dest='scat_file_path',
                        help='Path to the scat file. Pay attention to the \
                        wavelength in which you calculate the scat file. \
                        Since the extinction will be defined by this wavelength.\
                        The .scat files are probably at: pyshdom/mie_tables/polydisperse/(*.scat)', 
                        metavar='scat_file_path', required=True)
    
     
    
    parser.add_argument('--txt', type=str,
                        dest='csv_txt',
                        help='path to csv (txt) file where the lwc and reff are stored',
                        metavar='csv_txt', required=True)   
    
    
    parser.add_argument('--output', type=str,
                        dest='out_mat',
                        help='path to the output .mat file',
                        metavar='out_mat', default='test_extinction.mat')
    
    opts = parser.parse_args()
    
    exists(opts.scat_file_path, 'scat_file_path not found!')
    exists(opts.csv_txt, 'csv_txt not found!')
    
    # Mie scattering for water droplets
    mie = shdom.MiePolydisperse()
    print("reading the scat table from: {}".format(opts.scat_file_path))
    mie.read_table(opts.scat_file_path)

    # Generate a Microphysical medium
    droplets = shdom.MicrophysicalScatterer()
    #print("loading the csv file from: {}".format(opts.csv_txt))
    droplets.load_from_csv(opts.csv_txt,veff=0.1)
                          
    droplets.add_mie(mie)

    # extract the extinction:
    extinction = mie.get_extinction(droplets.lwc,droplets.reff,droplets.veff)
    extinction_data = extinction.data # 1/km
    dx = extinction.grid.dx
    dy = extinction.grid.dy
    
    nz = extinction.grid.nz
    Lz = extinction.grid.zmax - extinction.grid.zmin
    dz = Lz/nz
    
    # save extintcion as mat file:
    sio.savemat(opts.out_mat, dict(dx=dx,dy=dy,dz=dz,data_beta=extinction_data))  
    print("saving the .mat file to: {}".format(opts.out_mat))
    print('finished')
    
    """
    summary:
    The mat file must include the fields (dx,dt,dz,data_beta)
    
    -fields:
    dx # in km
    dy
    dz 
    data_beta # in 1/km it is the extinction.
    
    - example how to load it:
    test_extinction_path = 'test_data//rico32x37x26.mat'
    matData = Utils.load_MAT_FILE(test_extinction_path)
    # convert from km to meter
    dx = 1e3*Utils.float_round(matData['dx'][0][0])
    dy = 1e3*Utils.float_round(matData['dy'][0][0])
    dz = 1e3*Utils.float_round(matData['dz'][0][0])
    volumefield = 1e-3*matData['data_beta']
    
    
    """    

if __name__ == '__main__':
    main()