import numpy as np
import shdom
from shdom import float_round, core

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
integration_Dlambda = 10
# h is the Planck's constant, c the speed of light,
h = 6.62607004e-34 #J*s is the Planck constant
c = 3.0e8 #m/s speed of litght
k = 1.38064852e-23 #J/K is the Boltzmann constant

# ----------------------------------------------------------
# ----------------------------------------------------------
# ----------------------------------------------------------

# parameters that should be changes for different spectral ranges:
VIS = False
if(VIS):
    
    H = 500#km
    pixel_footprint = 0.02#km
    #orbital_speed = 7.61 # km/sec
    orbital_speed = 7.728 # km/sec
    scene_spectrum = [620,670]#nm
    pixel_size = 5.5 # microns
    QE = 0.45
    sun_zenith = 155 # deg
    sensor_zenith = 46 # deg
    trans = 0.88 # unitless transmittion.
    rho=0.3 # unitless cloud reflection.
    Imager_EFFICIANCY = 0.9 # unitless 
    READ_NOISE = 13 # electrons (RMS)
    DARK_NOISE = 125 # electrons/sec.
    sensor_bits = 10
    #--------------------------------
    required_well = 13.5e3
    full_well = 13.5e3
    #-------------------------------    

else: # SWIR is simulated
    H = 500#km
    pixel_footprint = 0.05#km
    #orbital_speed = 7.61 # km/sec
    orbital_speed = 7.728 # km/sec
    scene_spectrum = [1628,1658]#nm
    pixel_size = 10 # microns
    QE = 0.75
    sun_zenith = 155 # deg
    sensor_zenith = 46 # deg
    trans = 0.96 # unitless transmittion.
    rho=0.15 # unitless cloud reflection.
    Imager_EFFICIANCY = 0.9 # unitless  
    READ_NOISE = 160 # electrons (RMS)
    DARK_NOISE = 33.1e3 # electrons/sec.
    sensor_bits = 10
    #--------------------------------
    required_well = 100e3
    full_well = 450e3
    #-------------------------------    
    


ADD_POLAIZER = False

# ------------------------------------
max_exposure_time = 1e6*0.5*pixel_footprint/orbital_speed # in micron sec
#max_exposure_time = 1e6*pixel_footprint/orbital_speed

# why 0.5* ? The signal motion blur should be under a half-GSD.
gamma_cam = (2**sensor_bits)/full_well
QUANTIZATION_NOISE_VARIANCE = 1/(12*gamma_cam)
DARK_NOISE = 1e-6*max_exposure_time*DARK_NOISE
TOTAL_NOISE = DARK_NOISE + READ_NOISE + (required_well)
NOISE_FLOOR = READ_NOISE + DARK_NOISE

print("----> Required well is {}[electrons]".format((required_well)))
print("----> Exposure bound is set to {}[micro sec]".format(float_round(max_exposure_time)))
print("----> Noise floor bound is set to {}[electrons]".format(float_round(NOISE_FLOOR)))
print("----> Total Noise bound is set to {}[electrons]".format(float_round(TOTAL_NOISE)))

diffraction_scalar = 1.22
Dtheta = pixel_footprint/H
focal_length = 1e-3*(H*pixel_size)/pixel_footprint # mm


step = 10# nm
start = scene_spectrum[0]
stop = scene_spectrum[1]
lambdas = np.linspace(start, stop, int(((stop-start)/step)+1))
# radiance calculation:
LTOA = 6.8e-5*1e-9*shdom.plank(1e-9*lambdas) # units fo W/(m^2 nm)
tao = -np.log(trans)*(np.abs(1/np.cos(np.deg2rad(180-sun_zenith))) + np.abs(1/np.cos(np.deg2rad(sensor_zenith))))
transmission = np.exp(-tao)
Cosain = np.cos(np.deg2rad((180-sun_zenith)))
LTOA = LTOA * Cosain
radiance = transmission*(rho*LTOA)/np.pi # it is of units W/(m^2 st nm)

lens_diameter_diffration = 1e-6*max(diffraction_scalar*lambdas)/Dtheta # in mm
  

# -SNR calculations-----:
# To avoid saturation by dark-current, a constraint on cy is:
cy = 1 - ((2*DARK_NOISE)/required_well)
signal = cy*required_well

print("----> Focal length is set to {}[mm]".format(float_round(focal_length)))
print("----> Reached well is {}[electrons]".format((signal)))


G1 = (1/(h*c))*max_exposure_time*(((np.pi)/4)/(focal_length**2))*(pixel_size**2)
G2 = np.trapz((radiance*Imager_EFFICIANCY*QE)*lambdas, x = lambdas)
G = 1e-21*G1*G2

lens_diameter = 1000*((signal/G)**0.5) # in mm 

GAMMA_lambda = 1e-12*np.pi*Imager_EFFICIANCY*((lens_diameter/(2*focal_length))**2)*QE*(1e-9*lambdas/(h*c))*(pixel_size**2)
INTEGRAL = np.trapz(GAMMA_lambda*radiance, x = lambdas)
test_signal = (1e-6*max_exposure_time)*INTEGRAL
print("----> Sanity check: recomputed signal is {}[electrons]".format((test_signal)))

# maybe the diffraction limits the diameter, so:
print("----> Diameter to satisfay SNR is {}[mm]".format((lens_diameter)))
lens_diameter = max(lens_diameter,lens_diameter_diffration)

# aggregated noise variance is :
wave_diffraction_diameter = 1e-3*max(2*diffraction_scalar*lambdas)*(focal_length/lens_diameter)# in mirco m
number_pixels_in_spot = (wave_diffraction_diameter/pixel_size)**2
if(number_pixels_in_spot<1):
    
    number_pixels_in_spot = 1
    
all_noise_var = (number_pixels_in_spot*signal + np.ceil(number_pixels_in_spot)*(DARK_NOISE + QUANTIZATION_NOISE_VARIANCE + 
                                       (READ_NOISE**2)))

SNR = (number_pixels_in_spot*signal)/np.sqrt(all_noise_var)
print("----> SNR is {}".format(float_round(SNR)))
print("There are {} pixels in this spot".format(float_round(number_pixels_in_spot)))
        
print("----> Final Diameter is {}[mm]".format(float_round(lens_diameter)))
print("----> To be outside the diffreaction limit, the lens diameter should be larger than {} mm".format(float_round(lens_diameter_diffration)) )               
print("----> Spot size because of the diffraction is changed to {}[micro m]".format(float_round(wave_diffraction_diameter)))                
