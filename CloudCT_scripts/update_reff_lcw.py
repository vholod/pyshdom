
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import shdom
from shdom import float_round

droplets = shdom.MicrophysicalScatterer()

reff = sio.loadmat('../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_672-672nm_active_sats_10_GSD_20m_and_SWIR_1600-1600nm_active_sats_10_GSD_20m_LES_cloud_field_BOMEX/logs/reff_and_lwc_only_active_sats_10_BOMEX_21600_54x53x35_o1.txt-30-Aug-2020-18:34:40/FINAL_3D_reff.mat')
lwc = sio.loadmat('../CloudCT_experiments/VIS_SWIR_NARROW_BANDS_VIS_672-672nm_active_sats_10_GSD_20m_and_SWIR_1600-1600nm_active_sats_10_GSD_20m_LES_cloud_field_BOMEX/logs/reff_and_lwc_only_active_sats_10_BOMEX_21600_54x53x35_o1.txt-30-Aug-2020-18:34:40/FINAL_3D_lwc.mat')
droplets.load_from_csv('../synthetic_cloud_fields/BOMEX/BOMEX_21600_54x53x35_o1.txt')

lwc = shdom.GridData(droplets.grid, lwc['lwc'])
reff = shdom.GridData(droplets.grid, reff['reff'])

new_droplets = shdom.MicrophysicalScatterer(lwc, reff, droplets.veff)
# set medium:
new_medium = shdom.Medium(droplets.grid)
# add a scatterer to the medium:
new_medium.add_scatterer(new_droplets, name='clouds')
# show the medium to be sure the it was loaded correctly:

# save the txt file:
new_droplets.save_to_csv('BOMEX_21600_54x53x35_o1_sim.txt', comment_line='a part of the data of Eshkol april 2020')


