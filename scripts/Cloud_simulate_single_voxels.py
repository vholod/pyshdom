
# import mayavi.mlab as mlab

import matplotlib.pyplot as plt
import scipy.io as sio
from shdom.CloudCT_Utils import *


##-----------------------------------------------------
## Define functions
##-----------------------------------------------------
def generate_measurements_single(wavelength, lwc, reff, view_zenith, view_azimuth,
                          solar_zenith, solar_azimuth, veff=0.1, n_jobs=30):
    """
    This function uses the render_polarization_toa script
    to render a grid of measurements across microphysical parameters.
    The medium is a 1x1x20km atmosphere with rayleigh scattering and a single cloudy voxel in the center.

    Parameters
    ----------
    wavelength: list or np.array
        a list or array of wavelengths
    lwc: list or np.array
        a list or array of liquid water content.
    reff: list or np.array
        a list or array of effective radii
    view_zenith: float, list or np.array
        a list or array of view zenith angles
    view_azimuth: float, list or np.array
        The view azimuth angles
    solar_zenith: float, list or np.array
        a list or array of solar zenith angles
    solar_azimuth: float, list or np.array
        The solar azimuth angles
    veff: float
        The effective variance. Default is 0.1.
    n_jobs: int
        The number of jobs to parallelize across.

    Returns
    -------
    stokes: np.array(shape=(num_stokes, num_pixels, num_wavelength, num_lwc, num_reff))
        A multi-dimensional array containing a grid of measurements.
    """
    from render_radiance_toa import RenderScript
    wavelength = [str(i) for i in np.atleast_1d(wavelength)]
    view_zenith = [str(i) for i in np.atleast_1d(view_zenith)]
    view_azimuth = [str(i) for i in np.atleast_1d(view_azimuth)]
    solar_zenith = np.atleast_1d(solar_zenith)
    solar_azimuth = np.atleast_1d(solar_azimuth)

    if len(view_zenith) != len(view_azimuth):
        if len(view_zenith) == 1 and len(view_azimuth) > 1:
            view_zenith = solar_zenith * len(view_azimuth)
        elif len(view_azimuth) == 1 and len(view_zenith) > 1:
            view_azimuth = view_azimuth * len(view_zenith)
        else:
            raise AttributeError('error dimensions for solar zenith or azimuth')

    if len(solar_zenith) != len(solar_azimuth):
        if len(solar_zenith) == 1 and len(solar_azimuth) > 1:
            solar_zenith = np.full_like(solar_azimuth, solar_zenith)
        elif len(solar_azimuth) == 1 and len(solar_zenith) > 1:
            solar_azimuth = np.full_like(solar_zenith, solar_azimuth)
        else:
            raise AttributeError('error dimensions for solar zenith or azimuth')

    reff = np.atleast_1d(reff)
    lwc = np.atleast_1d(lwc)

    render_script = RenderScript()
    sys.argv = ['', ''] + wavelength + ['--generator', 'SingleVoxel'] + ['--x_res', '0.02'] + \
               ['--y_res', '0.02'] + ['--domain_size', '1.0', '--nx', '5', '--ny', '5'] + \
               ['--azimuth'] + view_azimuth + ['--zenith'] + view_zenith + \
               ['--n_jobs', str(n_jobs), '--veff', str(veff)] + ['--add_rayleigh']
    render_script.parse_arguments()
    rte_solvers = shdom.RteSolverArray()
    render_script.args.reff = reff
    render_script.args.lwc = lwc
    render_script.args.solar_zenith = solar_zenith
    render_script.args.solar_azimuth = solar_azimuth
    medium = render_script.get_medium()
    rte_solvers.add_solver(render_script.get_solver(medium))
    rte_solvers.solve(maxiter=100, verbose=True)
    measurements = render_script.render(medium.get_scatterer('cloud').bounding_box, rte_solvers)
    pixels = measurements.pixels
    return pixels

def loss(true_I, other_I):
    loss = np.linalg.norm(
        true_I - other_I, ord=2)
    return loss
'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

# This import registers the 3D projection, but is otherwise unused.

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



# -----------------------------------------------------------------
# -----------------------------------------------------------------


"""
Define AirMSPI zenith angles and polarization bands, flight direction, solar angles
"""
'''

view_zenith_list =[-15.0, -30.0] #5 cameras

view_azimuth_list =[-90.0, -90.0]
wavelengths_list = [[1.62]]

# Solar parameters
solar_azimuth = [0]#, 22.5, 45, 67.5, 90]
solar_zenith = [165]#165
'''


view_zenith_list = [0.0, 15.0, 30.0]


view_azimuth_list =[0.0, 90.0, 90.0]
wavelengths_list = [[0.66],[1.61]]

# Solar parameters
solar_azimuth = [0]#, 22.5, 45, 67.5, 90]
solar_zenith = [165]#165


"""
This main polarization rendering procedure 
It takes ~1.75 hours for 50x50 (lwc, reff) grid and n_jobs=30

Notes
-----
This cell should be ran with POLARIZATION compilation flag set to True
"""
# safe creation of a directory to save results
directory = 'single_view'
if not os.path.exists(directory):
    os.makedirs(directory)

# Microphysical grid definition
rmin = 2.0
rmax = 12.5

lmin = 1e-2
lmax = 0.85
options = 50
name = '50_50_angles_and_waves_r8_'
reff_range = np.linspace(rmin, rmax , options)
lwc_range = np.linspace(lmin, lmax, options)

# Run and save the results
lwc_true = [0.356]
#reff_true = [12.34]
reff_true = [8.0]
count = 0
for wavelengths in wavelengths_list:
    num_wavelengths = len(wavelengths)
    for view_zenith, view_azimuth in zip(view_zenith_list,view_azimuth_list):
        count += 1
        I_true = generate_measurements_single(wavelengths, lwc_true, reff_true,
                                            view_zenith, view_azimuth, solar_zenith, solar_azimuth)
        r = 0
        loss_I=np.zeros((len(reff_range),len(lwc_range)))
        for reff in reff_range:
            l = 0
            for lwc in lwc_range:
                I_others = generate_measurements_single(wavelengths, lwc, reff,
                                               view_zenith, view_azimuth, solar_zenith, solar_azimuth)
                loss_I[r,l] = loss(I_true,I_others)
                l += 1
            r += 1


        plt.subplot(2,5,count)
        x, y = np.meshgrid(lwc_range, reff_range, indexing='ij')
        a_dict = {'lwc':x, 'reff':y, 'loss':loss_I}
        view_zenith_str = str(view_zenith)
        if num_wavelengths>1:
            wavelengths_str='both_'+str(wavelengths[1])
        else:
            wavelengths_str = str(wavelengths)
        sio.savemat(directory+"/"+name+wavelengths_str+'_'+view_zenith_str+'.mat',{'a_dict':a_dict})
        c = plt.contourf(x,y,np.log(loss_I).T, cmap=plt.get_cmap('jet'), levels=15)
        cb = plt.colorbar(c)
    #plt.xlabel('lwc [g/kg]')
    #plt.ylabel('Reff [micron]')
#plt.savefig(directory + '/'+name+'.pdf')

'''
fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(x, y, np.log(loss_I).T,cmap='viridis', edgecolor='none')
a = plt.contourf(x,y,np.log(loss_I).T,15)
plt.xlabel('lwc [g/kg]')
plt.ylabel('Reff [micron]')
#a.set_xlim(lmin, lmax)
#a.set_ylim(rmin, rmax)
plt.savefig(directory + '/1600_10_10_surf.png')
'''

plt.show(block=True)

print('DONE Loss calculation')