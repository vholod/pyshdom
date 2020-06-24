import os 
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import shdom 
from shdom import float_round, core
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
import json
from shdom.CloudCT_Utils import *
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import OrderedDict
from math import log, exp, tan, atan, acos, pi, ceil, atan2
# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
integration_Dlambda = 10
# h is the Planck's constant, c the speed of light,
h = 6.62607004e-34 #J*s is the Planck constant
c = 3.0e8 #m/s speed of litght
k = 1.38064852e-23 #J/K is the Boltzmann constant


"""
This packege helps to define the imager to be simulated.
First, you need to define the sensor with its parameters. These parameters should be in the specs or given by by the companies.
Seconed, you neet to define the lens with its parameters. These parameters should be in the specs or given by by the companies.

Both the lens and the sensor can use efficiencies tables in csv format.
Sensor efficiency is QE - quantum effciency.
Lens efficienc is its TRANSMISSION.

After the defination of Sensor and Lens objects, you need to define an Imager object.
Imager object takes the following inputs:
* Sensor
* Lens
* scene spectrum -  which define only the range e.g. scene_spectrum=[400,1700].

The imager object can be set with uniform (simple model) QE and lens transtission. It is done for example by:
imager.assume_sensor_QE(45)
imager.assume_lens_UNITY_TRANSMISSION()
The important step here is to call the update method to apdate the imager with the new parameters e.g. imager.update().


Than, the Imager object can be used.
The recomended usage is as following (1-2 are needed for the rendering, the rest are aslo needed for radiometric manipulations):
1. Set Imager altitude (e.g imager.set_Imager_altitude(H=500) # in km).
2. Calculate the Imager footprint at nadir (e.g. imager.get_footprints_at_nadir()). The footprint is needed for the rendering simulation.
3. Get maximum exposur time. It is the exposure in which the motion blur would be less than 1 pixel (e.g. imager.max_esposure_time).
The maximum exposur time derivied only from the pixel footprint and orbital speed considerations. Currently the footprint is calculated for the nadir view.


TOADD...

"""


class SensorFPA(object):
    
    def __init__(self, QE=None, PIXEL_SIZE = 5.5 , FULLWELL = None, CHeight = 1000, CWidth = 1000, SENSOR_ID = '0',
                 READOUT_NOISE = 100, DARK_CURRENT_NOISE = 10, BitDepth = 8, TYPE = 'VIS'):
        """
        Parameters:
        QE - quantum effciency, which measures the prob-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the quantum effciency in range [0,1].
        PIXEL_SIZE: float
            it is the pixel pitch in microns. It assumses symetric pitch.
        FULLWELL: int
            pixel full well, how much eletrons a pixel can generate befor it saturates.
        CHeight: int
            Number of pixels in camera y axis
        CWidth: int
            Number of pixels in camera x axis
        SENSOR_ID: string
            It is possible to set an ID to object.
        READOUT_NOISE: int
            Sensor readout noise in units of [electrons]. It is random noise which does not depend on exposure time. It is due to read 
            circuits and output stages which add a random fluctuation to the output voltage. 
        DARK_CURRENT_NOISE: float
            Dark current shot noise in units of [electrons/(sec*temperatur)]. This is the output signal of 
            the device with no ambient illumination. It consists of thermally generated electrons within 
            the sensor which are accumulated during signal integration (self._Exposure time). 
            Dark signal is a function of temperature.
        BitDepth: int
            bit depth of the sensor.
        TYPE: str
            'VIS' or 'SWIR'.
        """
        
        self._SENSOR_ID = SENSOR_ID 
        self._CWidth = CWidth
        self._CHeight = CHeight
        self._PIXEL_SIZE = PIXEL_SIZE # PIXEL_SIZE im microns
        
        self._SENSOR_SIZE = np.array([CHeight,CWidth])*self._PIXEL_SIZE
        
        self._Exposure = 10 # mirco sec
        self._Gain = 0 # 0 means no gain.
        
        self._QE = QE
        if(QE is None):
            
            self._SPECTRUM = [] # will be defined when the QE will be given. SPECTRUM: list of 2 elements (floats) [lambda1, lambda2] in nm.
        else:
            assert isinstance(QE, pd.DataFrame), "QE must be pandas dataframe!"
            
            self._SPECTRUM = QE['<wavelength [nm]>'].values
            
        self._FULLWELL = FULLWELL
        # A common Rule of thumb is Full "well " ~ 1000 p^2, where p^2 is a pixel area in microns^2.
        if FULLWELL is None:
            
            if(TYPE is 'VIS'):
                self._FULLWELL = 1000*self._PIXEL_SIZE**2 # e-
            elif(TYPE is 'SWIR'):
                self._FULLWELL = (self._PIXEL_SIZE/15)*12e6 # e-
                # I based on Goldeye camera of alline vision, Saturation capacity -> 1.2 Me- <- (Gain0), 25 ke- (Gain2)
            else:
                raise Exception("Unknown sensor type")
            
            self._SNR = np.sqrt(self._FULLWELL)
            
        else:
            self._SNR = np.sqrt(self._FULLWELL)
            
        # Sensor's SNR is not equal to Imager SNR sinse the SNR of a siglan also depends on the signal itself.
        
        self._READOUT_NOISE = READOUT_NOISE
        self._DARK_CURRENT_NOISE = DARK_CURRENT_NOISE
        self._NOISE_FLOOR = None
        self._DR = None
        # self._NOISE_FLOOR = self._READOUT_NOISE + self._DARK_CURRENT_NOISE*(exposure*temp) , will be calculated later.
        # self._DR = self._FULLWELL/self._NOISE_FLOOR, will be calculated later.
        
        # Noise floor of the camera contains: self._READOUT_NOISE and self._DARK_CURRENT_NOISE.
        # Noise floor increases with the sensitivity setting of the sensor e.g. gain, exposure time and sensors temperature.
        # The Noise floor is important to define the Dynamic range (DN) of the sensor.
        # Dynamic range is defined as the ratio of the largest signal that an image sensor can handle to the 
        # readout noise of the camera.  Readout noise of the camera can be classified a "Dark Noise"  
        # which is measured during dark recording in specific temperature (e.g. room temp.) 
        # thus the Dark noise has a term of dark current shot noise. 
         
        self.BitDepth = BitDepth # bit depth of the sensor.
        self._TYPE = TYPE
        
    def Load_QE_table(self,csv_table_path = None):
        """
        Load QE from scv file "csv_table_path".
        QE - quantum effciency, which measures the prob-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the quantum effciency in range [0,1].
        
        Notes
        -----
        CSV format should be as follows:
        <wavelength [nm]>	<Efficiency>
        lambda1	         efficiency1
        .                    .
        .                    .
        """
        assert csv_table_path is not None, "You must provide QE table file"
        self._QE = pd.read_csv(csv_table_path)
        self._QE[self._QE<0] = 0 # if mistakly there are negative values. 
        self._QE['<Efficiency>'] = self._QE['<Efficiency>']/100  # effciency in range [0,1].
        self._SPECTRUM = self._QE['<wavelength [nm]>'].values
    
    def assume_QE(self,QE,spectrum):
        """
        Some times we don't know the QE or it is given for the whole spectrum as one number, so use that number here.
        """
        start = spectrum[0]
        stop = spectrum[1]        
        spectrum = np.linspace(start, stop, 4)        
        self._SPECTRUM = np.array(spectrum)
        df = pd.DataFrame(data={"<wavelength [nm]>":self._SPECTRUM,'<Efficiency>':(QE/100)*np.ones_like(self._SPECTRUM)},index=None)
        self._QE = df
        
    
        
    @property
    def bits(self):   
        """
        Retunrs the BitDepth of a pixel.
        """        
        return self.BitDepth
    
    @property
    def spectrum(self):    
        return self._SPECTRUM
    @property
    def QE(self):
        """
        Quantom Efficiency in range [0,1]
        """
        return self._QE  
    
    @QE.setter
    def QE(self,val):
        self._QE = val  
     
    @property
    def full_well(self):
        return self._FULLWELL    
        
    @full_well.setter
    def full_well(self,val):
        self._FULLWELL  = val      
      
    def set_DN_IN_DB(self,DR):
        self._DR = 10**(DR/20)
    def get_DR_IN_DB(self):
        return 20*np.log10(self._DR)  # in DB  
    def set_SNR_IN_DB(self,SNR):
        self._SNR = 10**(SNR/20)
    def get_SNR_IN_DB(self):
        return 20*np.log10(self._SNR)  # in DB   
    def set_exposure_time(self,time):
        """
        time must be in microns
        """        
        self._Exposure = time # mirco sec
    def set_gain(self,gain):
        assert gain >0, "gain must be positive."
        self._Gain = gain # 0 means no gain.  
        
        
    @property
    def pixel_size(self):    
        return self._PIXEL_SIZE
    
    @property
    def sensor_type(self):    
        return self._TYPE 
    
    @property
    def DynamicRange(self):    
        return self._DR
    
    @property
    def SNR(self):    
        return self._SNR 
    
    @SNR.setter
    def SNR(self,val):    
        self._SNR = val
    @property
    def sensor_size(self):   
        """
        Retunrs the sensor size. it is np.array with 2 elements relative to [H,W]
        """
        return self._SENSOR_SIZE
    
    @sensor_size.setter
    def sensor_size(self,val):   
        """
        Set the sensor size. it is np.array with 2 elements relative to [H,W]
        """
        self._SENSOR_SIZE = val  

    
    def get_sensor_resolution_in_lp_per_mm(self):
        """
        The resolution of the sensor, also referred to as the image space resolution for the system, 
        can be calculated by multiplying the pixel size in um by 2 (to create a pair), and dividing that into 1000 to convert to mm.
        The highest frequency which can be resolved by a sensor, the Nyquist frequency, is effectively two pixels or one line pair.
        https://www.edmundoptics.com/knowledge-center/application-notes/imaging/resolution/
        """
        return 1000/(2*self._PIXEL_SIZE)


class LensSimple(object):
    
    def __init__(self, TRANSMISSION=None, FOCAL_LENGTH = 100.0 , DIAMETER = 10.0 , LENS_ID = '0'):
        """
        Parameters:
        TRANSMISSION - measures the transmittion of the lens as a function of wavelength-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the transmittion [0,1].       
        FOCAL_LENGTH: floate
            The focaal length of the lens in mm.
        DIAMETER: floate
            The diameter of the lens in mm.
        LENS_ID: string
            It is possible to set an ID to object.
        """ 
        self._TRANSMISSION = TRANSMISSION
        if(TRANSMISSION is None):
            
            self._SPECTRUM = [] # will be defined when the QE will be given. SPECTRUM: list of 2 elements (floats) [lambda1, lambda2] in nm.
        else:
            assert isinstance(TRANSMISSION, pd.DataFrame), "TRANSMISSION must be pandas dataframe!"
            self._SPECTRUM = TRANSMISSION['<wavelength [nm]>'].values
            
        if FOCAL_LENGTH is not None:
            self._FOCAL_LENGTH = FOCAL_LENGTH # mm
        else:
            self._FOCAL_LENGTH = None
            
        if DIAMETER is not None:
            self._DIAMETER = DIAMETER # mm
        else:
            self._DIAMETER = None
        self._wave_diffraction = None # in mocro meters, it will be the 1e-3*min(2.44*self._SPECTRUM)*(self._FOCAL_LENGTH/self._DIAMETER)

        
       
    def Load_TRANSMISSION_table(self,csv_table_path = None):
        """
        TRANSMISSION - measures the transmittion of the lens as a function of wavelength-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the transmittion [0,1].
        
        Notes
        -----
        CSV format should be as follows:
        <wavelength [nm]>	<Efficiency>
        lambda1	         efficiency1
        .                    .
        .                    .
        """
        assert csv_table_path is not None, "You must provide QE table file"
        self._TRANSMISSION = pd.read_csv(csv_table_path)
        self._TRANSMISSION[self._TRANSMISSION<0] = 0 # if mistakly there are negative values. 
        self._TRANSMISSION['<Efficiency>'] = self._TRANSMISSION['<Efficiency>']/100  # effciency in range [0,1].
        self._SPECTRUM = self._TRANSMISSION['<wavelength [nm]>'].values
        self._wave_diffraction = 1e-3*min(2.44*self._SPECTRUM)*(self._FOCAL_LENGTH/self._DIAMETER)
        # https://www.edmundoptics.com/knowledge-center/application-notes/imaging/limitations-on-resolution-and-contrast-the-airy-disk/
        print("----> Spot size becous of the diffraction is {}[micro m]".format(float_round(self._wave_diffraction)))
    
    def assume_UNITY_TRANSMISSION(self,spectrum):
        """
        Some times we don't know the lens transmmision or don't want to set lens at the begining.
        So here, we just assume transmmision of 1 or maybe 0.99 (alfa).
        """
        alfa = 0.96
        start = spectrum[0]
        stop = spectrum[1]        
        spectrum = np.linspace(start, stop, 4)
        self._SPECTRUM = np.array(spectrum) # need to be more then 4 elements for the InterpolatedUnivariateSpline function in the future.
        df = pd.DataFrame(data={"<wavelength [nm]>":self._SPECTRUM,'<Efficiency>':alfa*np.ones_like(self._SPECTRUM)},index=None)
        self._TRANSMISSION = df
        
        
    @property
    def spectrum(self):    
        return self._SPECTRUM  
    @property
    def T(self):
        """
        Lens TRANSMISSION in range [0,1].
        """
        if self._TRANSMISSION is not None:
            return self._TRANSMISSION 
        else:    
            print("You did not set any lens transmission. A default of 1 is being used!.")
            return 1
    
    @T.setter
    def T(self,val):
        self._TRANSMISSION = val  
    
    @property
    def diameter(self):
        return self._DIAMETER
    
    @diameter.setter
    def diameter(self,val):
        self._DIAMETER = val      
    
        
    @property
    def focal_length(self):
        return self._FOCAL_LENGTH    

class Imager(object):
    def __init__(self, sensor=None,lens=None,scene_spectrum=[400,1700],integration_Dlambda = 10,temp=20,system_efficiency=1):
        """
        Parameters:
        sensor - sensor class.       
        lens - lens class.
        scene_spectrum: 2 element list
            list of 2 elements (floats) [lambda1, lambda2] in nm.
            The scene_spectrum will define the spectrums of QE of sensor and TRANSMMITION of the lens.
        integration_Dlambda: float
            It is the Dlambda that will be used in intgration and the delta in the interpoolations.
        temp: float
             temperatue in celsius. 
        system_efficiency: float
             range in [0,1], it is the camera system efficiency due to optics losses and sensor reflection (it is not a part of QE).
        
        """         
        if sensor is not None:
            self._SENSOR_DEFINED = True # means we have here defined sensor
        else:
            self._SENSOR_DEFINED = False
            
        if lens is not None:
            self._LENS_DEFINED = True # means we have here defined lens
        else:
            self._LENS_DEFINED = False
                
        self._sensor = sensor
        self._lens = lens
        self._temp = temp # celsius
        
        self._pixel_footprint = None # km, will be defined by function set_Imager_altitude()
        self._camera_footprint = None # km, will be defined by function set_Imager_altitude()
        
        if(self._SENSOR_DEFINED):# means we have here defined sensor
            self._DR = self._sensor.DynamicRange #dynamic range.
            self._SNR = self._sensor.SNR 
            self._DARK_NOISE = self._sensor._DARK_CURRENT_NOISE
            self._READ_NOISE = self._sensor._READOUT_NOISE
            self._NOISE_FLOOR = self._sensor._NOISE_FLOOR
            # self._NOISE_FLOOR = self._READOUT_NOISE + self._DARK_CURRENT_NOISE*(exposure*temp) , will be calculated later.
            # self._DR = self._FULLWELL/self._NOISE_FLOOR, will be calculated later.            
        else:
            self._DR = None #dynamic range.
            self._SNR = None 
            self._Imager_QE = None # it is not as self._sensor.QE
            # self._Imager_QE it is similar to self._sensor.QE but interpooleted on the full self._scene_spectrum.
            # it wii be set in the function self._adjust_spectrum()
        
        
            
        self._ETA = system_efficiency # It is the camera system efficiency, I don't know yet how to set its value, 
        if(self._LENS_DEFINED):# means we have here defined lens
            self._Imager_EFFICIANCY = self._ETA
        else:
            self._Imager_EFFICIANCY = None
            
            
        self._scene_spectrum = scene_spectrum
        self._scene_spectrum_in_microns = [float_round(1e-3*w) for w in self._scene_spectrum]
        self._integration_Dlambda = integration_Dlambda
        if(self._scene_spectrum is not None):
            integration_Dlambda = self._integration_Dlambda
            start = self._scene_spectrum[0]
            stop = self._scene_spectrum[1]
            step = integration_Dlambda # nm
            self._lambdas = np.linspace(start, stop, int(((stop-start)/step) + 1)) 
            self._centeral_wavelength = float_round(core.get_center_wavelen(start,stop))
            self._centeral_wavelength_in_microns = float_round(core.get_center_wavelen(self._scene_spectrum_in_microns[0],self._scene_spectrum_in_microns[1]))
            # self._centeral_wavelength will be relevant when we use spectrally-averaged atmospheric parameters.
        else:
            self._lambdas = None
            self._centeral_wavelength = None
            self._centeral_wavelength_in_microns = None
            # The self._centeral_wavelength_in_microns can be different from 1e-3*self._centeral_wavelength since the core.get_center_wavelen function depends on the order of the spectrum values.
        
        
        if(self._LENS_DEFINED and self._SENSOR_DEFINED):
            
            if self._sensor.QE is not None:        
                # if we here, the spectrums of sensor and maybe lens are defined and so are the QE and transmission.
                # the spectrum of the lens may still be empty, but its transmission is default of 1.
                assert(self._SENSOR_DEFINED), "Dad use of this class!, At this point at least the sensor should be defined."
                    
                self._adjust_spectrum()# updates self._lambdas, self._Imager_EFFICIANCY and self._Imager_QE
                
        else:
            print("Remember to set sensor QE and lens TRANSMISSION!")
            
        self._H = None # km, will be defined by function set_Imager_altitude()
        self._orbital_speed = None # [km/sec], will be defined by function set_Imager_altitude()
        self._max_esposure_time = None # [micro sec], will be defined by function set_Imager_altitude()
        self._FOV = None # [rad], will be defined by function set_Imager_altitude()  
        
        """
        To know how raniance converted to electrons we define GAMMA.
        The GAMMA depends on the wavelength. 
        The number of electrons i_e generated by photons at wavelength lambda during exposure time Dt is
        i_e = GAMMA_lambda * I_lambda * Dt,
        where I_lambda is the radiance [W/m^2 Sr] that reachs a pixel.
        GAMMA_lambda = pi*eta*((D/(2*f))^2)*QE_lambda * (lambda/(h*c))*p^2
        * p - pixel sixe.
        * h- Planck's constant, c - speed of light.
        * D -lens diameter, f- focal length.
        * eta - camera system efficiency due to optics losses and sensor reflection (it is not a part of QE).
        The units of GAMMA_lambda are [electrons * m^2 * Sr /joule ].
        """
        
        # primitive radiance calculation, later consider pyshdom:
        self._radiance = None # It is the I_lambda, the radiance that reach the imager per wavelength.
        """
        We can calculate I_lambda for wavelengths range since the calculation of pixel responce to light requires 
        an integral over solar spectral band. Thus RT simulations should be applied multiple times to calculated.
        But we there is an alternative:
        An alternative way is to use spectrally-averaged quantities, it is valid when wavelength dependencies within a
        spectral band are weak (e.g. in narrow band and absence absorption within the band).
        This alternative uses only one run of RT simulation per spectral band.
        So here, we define self.of_unity_flux_radiance. It is the radiance at the lens which calculated with RT 
        simulation when the solar irradiance at the TOA is 1 [W/m^2] and the spectrally-dependent parameters of the atmospheric model are spectrally-averaged.
        
        """
        
        self._of_unity_flux_radiance = None
        self._LTOA = None # irradiance on the top of Atmosphere.
    
    def update_minimum_lens_diameter(self):
        """
        Use this method if you constrained evrey thing and you want to calculate the minimum led diameter
        such that the pixel will reache its full well:
        """
        G1 = (1/(h*c))*self._max_esposure_time*(((np.pi)/4)/(self._lens._FOCAL_LENGTH**2))*(self._sensor.pixel_size**2)
        G2 = np.trapz((self._radiance*self._Imager_EFFICIANCY*self._Imager_QE)*self._lambdas, x = self._lambdas)
        G = 1e-21*G1*G2
        
        self._lens.diameter = 1000*((self._sensor.full_well/G)**0.5) # in mmm     
        print("\nUpdate minimum lens diameter")
        print("The diameter of the lens will change in this step, Is that what do you want to do?")
        print("----> Diameter is changed to {}[mm]".format(float_round(self._lens.diameter)))
        
        return self._lens.diameter
   
    def set_lens_diameter(self,diameter):
        """
        set lens diameter [mm] and update the SNR and the dymnamic range. 
        It still takes the exposure time to extrim and let the SNR and the dymnamic range to change.
        """

        G1 = (1/(h*c))*self._max_esposure_time*(((np.pi)/4)/(self._lens._FOCAL_LENGTH**2))*(self._sensor.pixel_size**2)
        G2 = np.trapz((self._radiance*self._Imager_EFFICIANCY*self._Imager_QE)*self._lambdas, x = self._lambdas)
        G = 1e-21*G1*G2
        
        assert diameter<(1000*(self._sensor.full_well/G)**0.5), "The diameter you set is not reasonble."
        
        self._lens.diameter = diameter
        signal = ((1e-3*self._lens.diameter)**2)*G
        #self._SNR = np.sqrt(signal)
        QUANTIZATION_NOISE = 0.5*(signal/(2**self._sensor.bits))
        DARK_NOISE = (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)
        self._SNR = (signal)/np.sqrt(signal + DARK_NOISE + (self._READ_NOISE**2) + (QUANTIZATION_NOISE**2))
        self._NOISE_FLOOR = self._READ_NOISE + DARK_NOISE            
        self._DR = signal/self._NOISE_FLOOR   
        
        print("\nUpdate Lens diameter")
        print("The diameter of the lens will change in this step, Is that what do you want to do?")
        print("----> Diameter is changed to {}[mm]".format(float_round(self._lens.diameter)))        
        print("----> Dynamic range is set to {} or {}[db]".format(float_round(self._DR),float_round(20*np.log10(self._DR))))
        print("----> SNR is set to {} or {}[db]".format(float_round(self._SNR),float_round(20*np.log10(self._SNR))))
        
        
    def set_imager_SRN(self,SNR):
        """ 
        set another SNR [not in db] not the sqrt of full well, but less, it is just to test how much the lens diameter can
        be decreased. It still takes the exposure time to extrim and let the diameter to change.
        """
        assert SNR<np.sqrt(self._sensor.full_well), "The SNR you set is not reasonble."
        
        G1 = (1/(h*c))*self._max_esposure_time*(((np.pi)/4)/(self._lens._FOCAL_LENGTH**2))*(self._sensor.pixel_size**2)
        G2 = np.trapz((self._radiance*self._Imager_EFFICIANCY*self._Imager_QE)*self._lambdas, x = self._lambdas)
        G = 1e-21*G1*G2
        #self._sensor.SNR = SNR I don't think to update the SNR of the sensor.
        self._SNR = SNR
        signal = SNR**2 # i need to ipdate thi later.
        
        
        self._lens.diameter = 1000*((signal/G)**0.5) # in mm
        self._NOISE_FLOOR = self._READ_NOISE + (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)            
        self._DR = signal/self._NOISE_FLOOR        
        print("\nUpdate imager SNR")
        print("The diameter of the lens will change in this step, Is that what do you want to do?")
        print("----> Diameter is changed to {}[mm]".format(float_round(self._lens.diameter)))        
        print("----> Dynamic range is set to {} or {}[db]".format(float_round(self._DR),float_round(20*np.log10(self._DR))))
        print("----> SNR is set to {} or {}[db]".format(float_round(self._SNR),float_round(20*np.log10(self._SNR))))
        
    
    def update_solar_angle(self,val):
        """
        To model the irradiance at a certain time of the day, we must
        multiple the irradiance at TOA by the cosine of the Sun zenith angle, it is also known as
        solar zenith angle (SZA) (1). Thus, the solar spectral irradiance at The TOA at
        a certain time is, self._LTOA = self._LTOA*cos(180-sun_zenith)
            
        input:
        val - is the zenith angle: float,
             Solar beam zenith angle in range (90,180]  
        """
        assert 90.0 < val <= 180.0, 'Solar zenith:{} is not in range (90, 180] (photon direction in degrees)'.format(val)
        Cosain = np.cos(np.deg2rad((180-val)))
        self._LTOA = self._LTOA * Cosain
        

    def update_scene_radiance_from_unity_solar_irradiance(self,val):
        """
        This method will update the radiance that reache the lens where the simulation of that radiance considered
        solar flux of 1 [W/m^2] and spectrally-dependent parameters of the atmospheric model are spectrally-averaged.
        
        """
        self._of_unity_flux_radiance = val
        
    def calculate_scene_radiance(self,rho=0.1,TYPE = 'simple'):
        """
        calculate the hypotetic radiance that would reach the imager lens.
        1. In the simple option, it is done very simple, just black body radiation and lamberation reflection be the clouds.
        The rho is the reflectance of the clouds (simple albedo).
        
        2. The best way to calculate the radiance is to use the pyshdom.
        So if TYPE = 'SHDOM', the self._radiance will be set to None and it will set to the right radiance after the rendering of pyshdom and the rigth radiometric mnipulations.
        
        """
        if(TYPE is 'simple'):
            self._LTOA = 6.8e-5*1e-9*shdom.plank(1e-9*self._lambdas) # units fo W/(m^2 nm)
            # I am assuming a solid angle of 6.8e-5 steradian for the source (the solar disk).
            self._radiance = (rho*self._LTOA)/np.pi # it is of units W/(m^2 st nm)
        
        elif(TYPE is 'SHDOM'):
            self._LTOA = 6.8e-5*1e-9*shdom.plank(1e-9*self._lambdas) # units fo W/(m^2 nm)
            # I am assuming a solid angle of 6.8e-5 steradian for the source (the solar disk).
            self._radiance = None # it is of units W/(m^2 st nm), in this case it will be set after the pyshdom rendering.
                
            
        else:
            
            raise Exception("Unsupported")
        
    def show_scene_irradiance(self): 
        """
        Shows the irradiance at the TOA:
        """
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.plot(self._lambdas,1000*self._LTOA ,label='black body radiation W/(m^2 um)')
        
        plt.ylim([0 ,1.1*max(1000*self._LTOA)])
        plt.xlim(self._scene_spectrum)
        
        plt.xlabel('wavelength [nm]', fontsize=16)
        plt.ylabel('Intensity', fontsize=16)
        plt.title('The irradiance at TOA')
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.grid(True)
        
    def show_scene_radiance(self, IFADD_IRRADIANCE = False):
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        if(IFADD_IRRADIANCE):
            plt.plot(self._lambdas,1000*self._LTOA ,label='black body radiation W/(m^2 um)')
        
        plt.plot(self._lambdas,1000*self._radiance,label='rediance on the lens W/(m^2 st um)')
        
        plt.ylim([0 ,1.1*max(1000*self._radiance)])
        plt.xlim(self._scene_spectrum)
        
        plt.xlabel('wavelength [nm]', fontsize=16)
        plt.ylabel('Intensity', fontsize=16)
        plt.title('The radiance at the imager')
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.grid(True)
        if(IFADD_IRRADIANCE):
            
            plt.legend()                
      
        
      
    def assume_sensor_QE(self,QE):
        """
        Some times we don't know the QE or it is given for the whole spectrum as one number, so use that number here.
        """
        self._sensor.assume_QE(QE,self._scene_spectrum)
        

    def assume_lens_UNITY_TRANSMISSION(self):
        """
        Some times we don't know the lens transmmision or don't want to set lens at the begining.
        So here, we just assume transmmision of 0.92-1.
        """
        if(not self._LENS_DEFINED):# means we no defined lens here
            self._lens = LensSimple(FOCAL_LENGTH = None , DIAMETER = None , LENS_ID = None)
        self._lens.assume_UNITY_TRANSMISSION(self._scene_spectrum)
        self._LENS_DEFINED = True
        print('Now the lens attribute in the imager instance is considered as defined.')

    def update(self):
        """
        Update should be called for example after the assumptions:
           imager.assume_sensor_QE()
           imager.assume_lens_TRANSMISSION()
           
           The update here similar to self._adjust_spectrum() but much simpler.
        """
        self._adjust_spectrum()      
        
    
    def update_sensor_size_with_number_of_pixels(self,nx,ny):
        """
        Set/update the sensor size by using new [nx,ny] resolution of an Imager. It can be done for instance, if the simulated resolution is smaller than the resolution from a spec.
        TODO - Be carfule here, this method doesn't update any other parameters.
        """
        self._sensor.sensor_size = np.array([nx,ny])*self._sensor.pixel_size
        
        # camera FOV:
        
        self._camera_footprint = 1e-3*(self._H*self._sensor.sensor_size)/self._lens._FOCAL_LENGTH #km
        # here self._camera_footprint is a np.array with 2 elements, relative to [H,W]. Which element to take?
        # currently I take the minimal volue:
        self._camera_footprint = max(self._camera_footprint)
        
        # Let self._camera_footprint be the footprint of the camera at nadir view in x axis. The field of view of the camera in radians is,
        self._FOV = 2*np.arctan(self._camera_footprint/(2*self._H))        
        
    
    def get_sensor_resolution(self):
        """
        Just get the [nx,ny] of the Imager's sensor.
        """
        return [int(i/self._sensor.pixel_size) for i in self._sensor.sensor_size]
    
        
    def _adjust_spectrum(self):
        """
        This function define common spectrum to sensor and lens and intarpoolats evrey thing into this spectrum.
        
        """
        
        lens_spectrum = self._lens.spectrum
        sensor_spectrum = self._sensor.spectrum   
        # Interpolate the Efficiencies along the wavelenghts axis:
        # Note:
        # The Efficiencies wavelengths are given in nm, and values are given in [0,1] range and not in percentage.        
        QE = InterpolatedUnivariateSpline(sensor_spectrum, self._sensor.QE['<Efficiency>'].values)
        QE_interpol = QE(self._lambdas)
        self._Imager_QE = QE_interpol
        
        if(len(lens_spectrum)==0):
            # if we here, it means the transmmission of the lens was not defined with some data and the 
            # trasmission is just 1.
            T_interpol = np.ones_like(self._lambdas)
        else:
            T = InterpolatedUnivariateSpline(lens_spectrum, self._lens.T['<Efficiency>'].values)
            T_interpol = T(self._lambdas)
        
        self._Imager_EFFICIANCY = self._ETA*T_interpol
        
        
    def show_efficiencies(self):
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.plot(self._lambdas,self._Imager_EFFICIANCY,linewidth=2,label='optics + sensor reflection')
        plt.plot(self._lambdas,self._Imager_QE,linewidth=2,label='sensor QE')
        total_efficeincy = self._Imager_EFFICIANCY*self._Imager_QE
        plt.plot(self._lambdas,total_efficeincy,linewidth=2,label='total efficiency')
        
        plt.ylim([0 ,1.1])
        plt.xlim(self._scene_spectrum)
        
        plt.xlabel('wavelength [nm]', fontsize=16)
        plt.ylabel('Efficiencies [unitless]', fontsize=16)
        plt.title('Efficiencies')
        
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.grid(True)
        plt.legend()        
    
        
    def set_Imager_altitude(self,H):
        """
        H must be in km
        """
        self._H = H # km
        self._orbital_speed = 1e-3*SatSpeed(orbit=self._H) # units of [m/sec] converted to [km/sec]
        print("----> Speed in {}[km] orbit is {}[km/sec]".format(self._H,float_round(self._orbital_speed)))
        if self._lens._FOCAL_LENGTH is not None:
            self.calculate_footprints()
        
            # bound the exposure time:
            # Let self._orbital_speed be the speed of the satellites. To avoid motion blur, it is important that:
            self._max_esposure_time = 1e6*self._pixel_footprint/self._orbital_speed # in micron sec
            
            # camera FOV:
            # Let self._camera_footprint be the footprint of the camera at nadir view in x axis. The field of view of the camera in radians is,
            self._FOV = 2*np.arctan(self._camera_footprint/2*self._H)
        
         
            self._NOISE_FLOOR = self._READ_NOISE + (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)
            self._DR = self._sensor.full_well/self._NOISE_FLOOR  
         
                 
    def set_radiance_at_whole_spectrum(self,val):
        """
        Set required radiance that should reach the lens.
        This method uses the focal length and expusur bound.
        This method calculates/updates:
        
        """
        pass
        
        
    def set_pixel_footprint(self,val):
        """
        Set the footprint of a pixel at nadir. This method calculates/updates the 
        focal length and expusur bound.
        """
        self._pixel_footprint = val
        self._lens._FOCAL_LENGTH = 1e-3*(self._H*self._sensor.pixel_size)/self._pixel_footprint # mm
        self._camera_footprint = 1e-3*(self._H*self._sensor.sensor_size)/self._lens._FOCAL_LENGTH #km
        # here self._camera_footprint is a np.array with 2 elements, relative to [H,W]. Which element to take?
        # currently I take the minimal volue:
        self._camera_footprint = max(self._camera_footprint)
        
        # bound the exposure time:
        # Let self._orbital_speed be the speed of the satellites. To avoid motion blur, it is important that:
        self._max_esposure_time = 1e6*self._pixel_footprint/self._orbital_speed # in micron sec
        self._NOISE_FLOOR = self._READ_NOISE + (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)
        self._DR = self._sensor.full_well/self._NOISE_FLOOR          
        # camera FOV:
        # Let self._camera_footprint be the footprint of the camera at nadir view in x axis. The field of view of the camera in radians is,
        self._FOV = 2*np.arctan(self._camera_footprint/2*self._H)
        
        # SNR:
        QUANTIZATION_NOISE = 0.5*(self._sensor.full_well/(2**self._sensor.bits))
        DARK_NOISE = (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)
        self._SNR = (self._sensor.full_well)/np.sqrt(self._sensor.full_well + DARK_NOISE + (self._READ_NOISE**2) + (QUANTIZATION_NOISE**2))        
        #self._SNR = (self._sensor.full_well)**0.5 old version
        print("The focal length will change in this step, This what do you want to do?")
        print("----> Focal length is set to {}[mm]".format(float_round(self._lens.focal_length)))
        print("----> Exposure bound is set to {}[micro sec]".format(float_round(self._max_esposure_time)))
        print("----> Dynamic range bound is set to {} or {}[db]".format(float_round(self._DR),float_round(20*np.log10(self._DR))))
        print("----> Noise floor bound is set to {}[electrons]".format(float_round(self._NOISE_FLOOR)))
        print("----> SNR is {} or {}[db]".format(float_round(self._SNR),float_round(self._sensor.get_SNR_IN_DB())))
        
        
    def calculate_footprints(self):
        """
        Calculats footprints in km:
        """
        self._pixel_footprint = 1e-3*(self._H*self._sensor.pixel_size)/self._lens._FOCAL_LENGTH #km
        self._camera_footprint = 1e-3*(self._H*self._sensor.sensor_size)/self._lens._FOCAL_LENGTH #km
        # here self._camera_footprint is a np.array with 2 elements, relative to [H,W]. Which element to take?
        # currently I take the minimal volue:
        self._camera_footprint = max(self._camera_footprint)
        
        
        
    def set_system_efficiency(self,val):
        """
        Set the camera system efficiency. The camera system efficiency is due to optics losses and sensor reflection (it is not a part of QE).
        """        
        assert val>1 or val<0 , "system efficiency must be in the [0,1] range."
        self._ETA = val
    
    def get_system_efficiency(self):
        return self._ETA       
        
    def get_footprints_at_nadir(self):
        """
        Get pixel footprint and camera footprint at nadir view only.
        """
        return self._pixel_footprint, self._camera_footprint
     
     
    @property 
    def max_esposure_time(self):
        return self._max_esposure_time
    
    
    @property 
    def centeral_wavelength_in_microns(self):
        return self._centeral_wavelength_in_microns

    @property 
    def scene_spectrum(self):
        return self._scene_spectrum
    
    @property 
    def max_noise_floor(self):
        return self._NOISE_FLOOR        
    
    @property 
    def min_dynamic_range(self):
        return self._DR   
    
    @property 
    def L_TOA(self):
        """
        returns the irradiance at the TOA
        """
        return self._LTOA
    
    @property 
    def FOV(self):
        """
        returns the Field of view of a camera.
        """
        return self._FOV # be carfule, it is in radiance.
    
    @classmethod
    def ImportConfig(cls,file_name = 'Imager_config.json'):
        """
        Import Imager configuration.
        """
        obj = cls.__new__(cls)  # Does not call __init__
        
        with open(file_name) as json_file:
            data = json.load(json_file)
            
        
        # Define sensor:
        QE = pd.DataFrame(data={"<wavelength [nm]>":data['SPECTRUM'],'<Efficiency>':data['SENSOR_QE']},index=None)
        
        sensor = SensorFPA(QE=QE,PIXEL_SIZE = data['PIXEL_SIZE'],
                           FULLWELL = data['FULLWELL'],
                           CHeight = data['CHeight'],
                           CWidth = data['CWidth'],
                           SENSOR_ID = '111',
                           READOUT_NOISE = data['READOUT_NOISE'],
                           DARK_CURRENT_NOISE = data['DARK_CURRENT_NOISE'],
                           BitDepth = data['BitDepth'],
                           TYPE = data['SENSOR_TYPE']) 
        
    
        # Define lens:  
        
        TRANSMISSION = pd.DataFrame(data={"<wavelength [nm]>":data['SPECTRUM'],'<Efficiency>':data['LENS_TRANSMISSION']},index=None)
        
        lens = LensSimple(TRANSMISSION= TRANSMISSION,
                         FOCAL_LENGTH = data['LENS_FOCAL_LENGTH'],
                         DIAMETER = data['LENS_DIAMETER'],
                         LENS_ID = '101')
       
             
        super(Imager, obj).__init__()  # Don't forget to call any polymorphic base class initializers
        
        
        obj._SENSOR_DEFINED = True # means we have here defined sensor
        obj._LENS_DEFINED = True # means we have here defined lens
        
        obj._sensor = sensor
        obj._lens = lens
        obj._temp = data['TEMPERTURE'] # celsius
    
        
        obj._DR = obj._sensor.DynamicRange #dynamic range.
        obj._SNR = obj._sensor.SNR 
        obj._DARK_NOISE = obj._sensor._DARK_CURRENT_NOISE
        obj._READ_NOISE = obj._sensor._READOUT_NOISE
        obj._NOISE_FLOOR = obj._sensor._NOISE_FLOOR
        
        obj._Imager_QE = obj._sensor.QE
        
        obj._ETA = data['SYSTEM_EFFICIENCY'] # It is the camera system efficiency, I don't know yet how to set its value, 
        obj._Imager_EFFICIANCY = obj._ETA
        
        obj._lambdas = obj._sensor.spectrum 
        obj._scene_spectrum = [data['START_SPECTRUM'], data['END_SPECTRUM']]
        obj._scene_spectrum_in_microns = [float_round(1e-3*w) for w in obj._scene_spectrum]
        obj._centeral_wavelength = float_round(core.get_center_wavelen(obj._scene_spectrum[0],obj._scene_spectrum[1]))   
        obj._centeral_wavelength_in_microns = float_round(core.get_center_wavelen(obj._scene_spectrum_in_microns[0],obj._scene_spectrum_in_microns[1]))
        obj._integration_Dlambda = integration_Dlambda
        
        obj._H = None 
        obj._orbital_speed = None 
        obj._max_esposure_time = None 
        obj._FOV = None       
        
        obj._radiance = None
        obj._LTOA = np.array(data['SPECTRAL_IRRADIANCE_TOA'])
        
        return obj
        
    def short_description(self):
        
        """
        returns a string with a short description of the Imager
        """
        return self._sensor.sensor_type
        
    def ExportConfig(self,file_name = 'Imager_config.json'):
        """
        report evrey thing in a table:
        """
        _DICT_ = OrderedDict()
        _DICT_['PIXEL_SIZE'] = self._sensor.pixel_size # microns
        _DICT_['FULLWELL'] = self._sensor.full_well  # electrons
        _DICT_['CHeight'] =  int(self._sensor.sensor_size[0]) if self._sensor.sensor_size[0] is not None else None# pixels
        _DICT_['CWidth'] =  int(self._sensor.sensor_size[1]) if self._sensor.sensor_size[1] is not None else None# pixels
        _DICT_['BitDepth'] = self._sensor.bits # unitless
        _DICT_['DYNAMIC_RANGE'] = self._sensor.DynamicRange # unitless, it is not in DB
        _DICT_['SNR'] = float_round(self._sensor.SNR) # unitless, it is not in DB
        _DICT_['DARK_CURRENT_NOISE'] = self._sensor._DARK_CURRENT_NOISE # [electrons/(sec*temperatur)]
        _DICT_['READOUT_NOISE'] = self._sensor._READOUT_NOISE # electrons
        _DICT_['TEMPERTURE'] = self._temp # celsius
        _DICT_['NOISE_FLOOR'] = float_round(self._NOISE_FLOOR) if self._NOISE_FLOOR is not None else None # electrons
        _DICT_['SYSTEM_EFFICIENCY'] = self._ETA # unitless
        _DICT_['MAX_EXPOSURE_TIME'] = int(self._max_esposure_time) if self._max_esposure_time is not None else None# in [micro sec]
        _DICT_['START_SPECTRUM'] = self._scene_spectrum[0] # in nm
        _DICT_['END_SPECTRUM'] = self._scene_spectrum[1] # in nm
        _DICT_['SPECTRUM'] = self._lambdas.tolist() if self._lambdas is not None else None # in nm
        _DICT_['SENSOR_QE'] = self._Imager_QE.tolist() if self._Imager_QE is not None else None # to update units.
        _DICT_['SENSOR_TYPE'] = self._sensor.sensor_type
        _DICT_['SPECTRAL_IRRADIANCE_TOA'] = self._LTOA.tolist() if self._LTOA is not None else None # in W/(m^2 st nm)
        _DICT_['LENS_DIAMETER'] = self._lens.diameter # in mm 
        _DICT_['LENS_FOCAL_LENGTH'] = self._lens.focal_length # in mm 
        T = self._Imager_EFFICIANCY/self._ETA if self._Imager_EFFICIANCY is not None else None
        _DICT_['SYSTEM_EFFICIENCY'] = self._ETA
        if(np.isscalar(T)):
            _DICT_['LENS_TRANSMISSION'] = T
        else:
            _DICT_['LENS_TRANSMISSION'] = T.tolist() # unitless
        
        #dumping a dictionary to a YAML file while preserving order
        # https://stackoverflow.com/a/8661021
        with open(file_name, 'w') as f:
            json.dump(_DICT_, f, indent=4)
        
        #with open(file_name, 'w') as file:
            #documents = yaml.dump(_DICT_, file_name)            
            
        #pass
        #COLUMNS = ['sensor id','Resolution','Spectral range','Sensor type','Pixel size','Read noise'
        #,'Dark current','Full well','Dynamic range','SNR','Bit depth']
        
        #data = {'sensor id':,'Resolution':,'Spectral range':,'Sensor type':,'Pixel size':,
                #'Read noise':,'Dark current':,'Full well':,'Dynamic range':,'SNR':,'Bit depth':,
                #'Orbit':,'Focal length':,'Exposure bound':,'Lens diameter'}
                
        
        assert (self._SENSOR_DEFINED and  self._LENS_DEFINED), "Too soon to report, define all needed"

        if(self._radiance is not None):
            
            G1 = (1/(h*c))*self._max_esposure_time*(((np.pi)/4)/(self._lens._FOCAL_LENGTH**2))*(self._sensor.pixel_size**2)
            G2 = np.trapz((self._radiance*self._Imager_EFFICIANCY*self._Imager_QE)*self._lambdas, x = self._lambdas)
            G = 1e-21*G1*G2
            
            signal = ((1e-3*self._lens.diameter)**2)*G
            self._NOISE_FLOOR = self._READ_NOISE + (self._DARK_NOISE*1e-6*self._max_esposure_time*self._temp)            
            self._DR = signal/self._NOISE_FLOOR   
        else:
            self._NOISE_FLOOR = None
            self._DR  = None
        
        print("\nReport all")
        print("----> Diameter is {}[mm]".format(float_round(self._lens.diameter))) 
        if(self._DR is not None):
            print("----> Dynamic range is {} or {}[db]".format(float_round(self._DR),float_round(20*np.log10(self._DR))))
        print("----> SNR is {} or {}[db]".format(float_round(self._SNR),float_round(20*np.log10(self._SNR))))
        print("----> Full_well is {}[electrons]".format(float_round(self._sensor.full_well)))
        if(self._radiance is not None):
            print("----> signal is {}[electrons]".format(float_round(signal)))
        
#---------------------------------------------------------------------
#-------------------------MAIN----------------------------------------
#---------------------------------------------------------------------
if __name__ == '__main__':
    """
    if you have the efficiency (QE for camera, TRANSMISSION for lens) in any type, convert it to csv file.
    # If its in image, use https://apps.automeris.io/wpd/ to extract point interactivly.
    # the format should be as following:
    <wavelength [nm]>	<Efficiency>
    lambda1	         efficiency1
    .                    .
    .                    .
    
    If you do not have certian efficiency description and you want to create your own, use function GenerateTable()
    """
    IF_GENERATE_TABLSE = False
    # meanwhile i am playing:
    scene_spectrum=[400,1700]
    
    LENS_TRANSMISSION_CSV_FILE = '../notebooks/example_lens_transmission.csv'
    SENSOR_QE_CSV_FILE = '../notebooks/example_sensorQE.csv'
    
    if(IF_GENERATE_TABLSE):
        # generate sensor QE:    
        GenerateTable(scene_spectrum=scene_spectrum,SAVE_RESOLT_AS = SENSOR_QE_CSV_FILE)
        # generate lens transmission:
        GenerateTable(scene_spectrum=scene_spectrum,SAVE_RESOLT_AS = LENS_TRANSMISSION_CSV_FILE)
    
    
    
    # Define sensor:
    sensor = SensorFPA(PIXEL_SIZE = 5.5 , CHeight = 2048, CWidth = 2048, SENSOR_ID = '0',
                     READOUT_NOISE = 100, DARK_CURRENT_NOISE = 10, BitDepth = 8)
    sensor.Load_QE_table(SENSOR_QE_CSV_FILE)    
    # Define lens:    
    lens = LensSimple(FOCAL_LENGTH = 95.0 , DIAMETER = 50.0 , LENS_ID = '0')
    lens.Load_TRANSMISSION_table(LENS_TRANSMISSION_CSV_FILE)    
    # create imager:
    imager = Imager(sensor=sensor,lens=lens,scene_spectrum=scene_spectrum)
    imager.show_efficiencies()
    
    # set geometry:
    H = 500 # km
    imager.set_Imager_altitude(H=H)
    pixel_footprint, camera_footprint = imager.get_footprints_at_nadir()
    max_esposure_time = imager.max_esposure_time
    pixel_footprint = float_round(pixel_footprint)
    camera_footprint = float_round(camera_footprint)
    max_esposure_time = float_round(max_esposure_time)
    print("At nadir:\n Pixel footprint is {}[km]\n Camera footprint is {}\n[km]\n Max esposure time {}[micro sec]\n"
          .format(pixel_footprint, camera_footprint, max_esposure_time))
    
    imager.ExportConfig(file_name = 'Imager_config.json')
    
    test_imager = Imager()
    test_imager.ImportConfig(file_name = 'Imager_config.json')
    test_imager.ExportConfig(file_name = 'test_Imager_config.json')
    
    plt.show()
    
