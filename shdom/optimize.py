"""
Optimization and related objects to monitor and log the optimization process.
"""
import numpy as np
import time, os, copy, shutil
from scipy.optimize import minimize, least_squares
from scipy.optimize import basinhopping
import scipy.io as sio
import copy
from shdom.rays_in_voxels import *

import shdom
from shdom import GridData, core, float_round
import dill as pickle
import itertools
from joblib import Parallel, delayed
from collections import OrderedDict
import tensorboardX as tb
import matplotlib.pyplot as plt
import warnings
from itertools import chain
from scipy import ndimage
from scipy.ndimage.filters import laplace as laplacian


class OpticalScattererDerivative(shdom.OpticalScatterer):
    """
    An OpticalScattererDerivative object.
    Essentially identical to a shdom.OpticalScatterer with no restrictions on negative extinction or albedo values outside of [0,1].

    Parameters
    ----------
    wavelength: float
        A wavelength in microns
    extinction: shdom.GridData object
        A GridData object containing the extinction (1/km) on a grid
    albedo: shdom.GridData
        A GridData object containing the single scattering albedo [0,1] on a grid
    phase: shdom.GridPhase
        A GridPhase object containing the phase function on a grid
    """
    def __init__(self, wavelength, extinction=None, albedo=None, phase=None):
        super().__init__(wavelength, extinction, albedo, phase)
        
    def resample(self, grid):
        """
        The resample method resamples the OpticalScatterer (extinction, albedo, phase).

        Parameters
        ----------
        grid: shdom.Grid
            The new grid to which the data will be resampled

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            An optical scatterer resampled onto the input grid
        """
        extinction = self.extinction.resample(grid)
        albedo = self.albedo.resample(grid)
        phase = self.phase.resample(grid)            
        return shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)
    
    @property
    def extinction(self):
        return self._extinction
    
    @extinction.setter
    def extinction(self, val):
        self._extinction = val
    
    @property
    def albedo(self):
        return self._albedo
    
    @albedo.setter
    def albedo(self, val):
        self._albedo = val
        

class GridPhaseEstimator(shdom.GridPhase):
    """
    A GridPhaseEstimator.

    Notes
    -----
    A dummy class, currently not implemented.
    """
    def __init__(self, legendre_table, index):
        super().__init__(legendre_table, index)


    
class TheoreticProfile(object):
    """
    Used to fit "good estimation" layerd (1D) profile to estimator.
    The "good estimation" is considered as reasonabal estimation to the microphysics.
    Parameters:
    parameters: int
       Number of the model parameters
    altitudes: np array
        array of altitudes
    data_fun: function
        The function that will be used in the fitting, the pure theoretic one.
    every_iter: - int
          It will be used in fitting a 1D profile every X iterations.
    ---------
    
    """ 
    def __init__(self, parameters, altitudes , data_fun , every_iter = 1e10, profile_name = 'reff'):
        self._data_fun = data_fun
        self._every_iter = every_iter
        self._N_parameters = parameters
        self._altitudes = altitudes
        self._init_parameters = np.ones([self._N_parameters])
        self._profile_name = profile_name
        
    def init_parameters(self,parameters_values):
        if(isinstance(parameters_values,list)):
            parameters_values = np.array(parameters_values)
        self._init_parameters = parameters_values
        self._N_parameters = parameters_values.size
        
    def tif_fun(self, x, z, y):
        """
        x - model parameters.
        """
        f = self._data_fun(x,z)
        return f - y
    
    def play_generate_noisy_data(self):
        """
        Only to play and test this class.
        """
        true_parameters = np.array([16,3,0.44]) # 20*np.random.rand(self._N_parametersne)
        noise = 1
        n_outliers = 5
        y = self._data_fun(true_parameters,self._altitudes)
        rnd = np.random.RandomState(0)
        error = noise * rnd.randn(self._altitudes.size)
        outliers = rnd.randint(0, self._altitudes.size, n_outliers)
        error[outliers] *= 3
        true_profil = y    
        test_profil = y + error
        return test_profil, true_profil
    
    def standard_least_squares(self,profil):
        """
        profil to fit to
        """
        res = least_squares(self.tif_fun, self._init_parameters, args=(self._altitudes,profil))
        y = self._data_fun(res.x, self._altitudes)
        return res.x, y

    def robust_least_squares(self,profil):
        """
        profil to fit to
        """
        res = least_squares(self.tif_fun, self._init_parameters, loss='soft_l1', f_scale=0.1, args=(self._altitudes,profil))
        y = self._data_fun(res.x, self._altitudes)
        return res.x, y    
    
    def IS_ITERATION_TO_FIT(self,iteration):
        if(iteration > 0 and (iteration % self._every_iter == 0)):
            return True
        else:
            return False
    
    @property
    def profile_name(self):
        return self._profile_name
    
    @property
    def altitudes(self):
        return self._altitudes    
                     
    @altitudes.setter
    def altitudes(self,val):
        self._altitudes = val
    
class GridDataEstimator(shdom.GridData):
    """
    A GridDataEstimator defines unknown shdom.GridData to be estimated.

    Parameters
    ----------
    grid_data: shdom.GridData
        The initial guess for the estimator
    min_bound: float, optional
        A lower bound for the parameter values
    max_bound: float, optional
        An upper bound for the parameter values
    """
    def __init__(self, grid_data, min_bound=None, max_bound=None, precondition_scale_factor=1.0):
        super().__init__(grid_data.grid, grid_data.data)
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._mask = None
        self._num_parameters = self.init_num_parameters()
        self._precondition_scale_factor = precondition_scale_factor
        self._theoretic_profile = None # in case of fitting estimated/theoretic 1d profile to the estimator.
        self._grad_scale = 1
        self._precond_grad = False
        
    def set_grad_scale(self, grid=None, mask=None, alpha_core=2, alpha_out=0):
        '''
        Set grad scale
        Returns
        -------

        '''

        if grid is None:
            grid = self.grid
            
        struct = ndimage.generate_binary_structure(3, 5)
        erodedMask = ndimage.binary_erosion(mask.data.astype(np.int16), structure=struct)
        
        scale = (alpha_core*np.multiply(erodedMask, 1, dtype=np.float32) +  alpha_out*np.multiply(mask, 1,
                                                                                                dtype=np.float32))/2
        
        if alpha_out>0:
            scale = ndimage.gaussian_filter(scale, sigma = 2)*mask
            
        if grid.type == '1D':
            self.grad_scale = np.squeeze(np.mean(np.mean(scale, axis=0), axis=0))
        elif grid.type == '3D':
            self.grad_scale = scale    
    
    def add_theoretic_profile(self,profile):
        """
        Adding theoretic profile (good estimation of the microphysics) to the estimator.
        It will be used in fitting a 1D profile every X iterations.
        """
        self._theoretic_profile = profile
        
    def set_state(self, state, optimization_iteration = None):
        """
        Set the estimator state.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The state to set the estimator data (grid is left unchanged)

        Notes
        -----
        The state is scaled back by the preconditioning scale factor
        If the estimator has a mask, data point outside of the mask are left uneffected.
        """
        state = state / self.precondition_scale_factor
        # Here the state already is scaled back.
        if self.mask is None:
            self._data = np.reshape(state, self.shape)
        else:
            self._data[self.mask.data] = state
            
            if(self._theoretic_profile is not None):
                assert optimization_iteration is not None, "To use profile fitting you must provied optimization iteration"
                if(self._theoretic_profile.IS_ITERATION_TO_FIT(optimization_iteration)):
                    
                    if(self.grid.type == '1D'):
                        data = self._data[self.mask.data]
                    elif(self.grid.type == '3D'):
                        masked_mean = np.ma.masked_equal(self._data, 0).mean(axis=0).mean(axis=0)
                        data = masked_mean.data   
                        raise Exception('Still unsupported for grid type 3D')
                    else:
                        raise Exception('To fit an estimated curve the grid must be either 1D or 3D')
                    
                    print("Iteration #{}: fitting estimated profile of {}".format(optimization_iteration,self._theoretic_profile.profile_name))
                    parameters, data = self._theoretic_profile.robust_least_squares(data)
                    self._data[self.mask.data] = data
                
            

    def get_state(self):
        """
        Retrieve the medium state.

        Returns
        -------
        preconditioned_state: np.array(dtype=np.float64)
            The preconditioned state scaled by a preconditioning scale factor.

        Notes
        -----
        If the estimator has a mask, data point outside of the mask are not retrieved.
        """
        if self.mask is None:
            data = self.data.ravel()
        else:
            data = self.data[self.mask.data]
        preconditioned_state = data * self.precondition_scale_factor
        return preconditioned_state

    def init_num_parameters(self):
        """
        Initialize the number of parameters to be estimated.

        Returns
        -------
        num_parameters: int
            The number of parameters to be estimated.

        Notes
        -----
        If the estimator has a mask, data point outside of the mask are not counted.
        """
        if self.mask is None:
            num_parameters = self.data.size
        else:
            num_parameters = np.count_nonzero(self.mask.data)
        return num_parameters

    def set_mask(self, mask):
        """
        Set a mask for GridDataEstimator data points and zero points outside of the masked region.

        Parameters
        ----------
        mask: shdom.GridData object
            A GridData object with boolean data. True is for data points that will be estimated.
        """
        self._mask = mask.resample(self.grid, method='nearest')
        self._num_parameters = self.init_num_parameters()
        super().apply_mask(self.mask)

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        min_bound = self.min_bound * self.precondition_scale_factor if self.min_bound is not None else None
        max_bound = self.max_bound * self.precondition_scale_factor if self.max_bound is not None else None
        return [(min_bound, max_bound)] * self.num_parameters

    def project_gradient(self, gradient, grid, scaling = None):
        """
        Project gradient onto the state representation.

        Parameters
        ----------
        grid: shdom.Grid
            The internal shdom grid upon which the gradient was computed.
        gradient: np.array(dtype=np.float64)
            An array containing the gradient of the cost function with respect to the parameters.
        scaling: None or np.array that scales the shdom.GridData
        Returns
        -------
        state_gradient: np.array(dtype=np.float64)
            State gradient representation
        """
        gradient = gradient.squeeze(axis=-1)
        state_gradient = shdom.GridData(grid, gradient, scaling).resample(self.grid)
        #state_gradient.data *= self._grad_scale
        
        # vadim try this after Roi suggestion:
        # state_gradient.data /= self.precondition_scale_factor
        scale_grad = self._precond_grad
        
        # ---- consider the above statment
        debug_gradient = state_gradient.data # vadim added to get the gradient in the base grid to debug/visualize it later on.
        if self.mask is None:
            if(scale_grad):
                return state_gradient.data.ravel()/self.precondition_scale_factor, debug_gradient
            else:
                return state_gradient.data.ravel(), debug_gradient
        else:
            debug_gradient[self.mask.data == False] = 0
            if(scale_grad):
                return state_gradient.data[self.mask.data]/self.precondition_scale_factor, debug_gradient
            else:
                return state_gradient.data[self.mask.data], debug_gradient
    
    def erode_edges(self,mask, min_val=0.0, erode_radius = 2, sigma = 0.15):
        """
        Erode and filter the data inside the mask.

        Parameters
        ----------
        mask: shdom.GridData object
            A GridData object with boolean data. True is for data points that will be estimated.
        erode_radius: int
            Erosion radius of the 3D mask.
        sigma: int
            Std of the gaussian to smooth the eroded mask.
        """
        if self.type == '3D':
            mask = mask.resample(self.grid, method='nearest')
            
            alpha_out = 1.5 #2 - for uniform
            alpha_core = 2-alpha_out #0 - for uniform
        
            struct = ndimage.generate_binary_structure(3, erode_radius)
            erodedMask = ndimage.binary_erosion(mask.data.astype(np.int16), structure=struct)
        
            scale = (alpha_core*np.multiply(erodedMask, 1, dtype=np.float32) +  alpha_out*np.multiply(mask.data, 1,
                                                                                                      dtype=np.float32))/2
            scale = ndimage.gaussian_filter(scale, sigma = sigma)*mask.data
            scale /= scale.max()
            
            self._data *= scale
            self._data[self._data<min_val] = min_val
            self._data[self._data<self._min_bound] = self._min_bound + 0.01*self._min_bound
            self._data[self._data>self._max_bound] = self._max_bound - 0.01*self._max_bound
            self._data[np.bitwise_not(mask.data)] = 0.0
            
    @property
    def precond_grad(self):
        return self._precond_grad
    
    @precond_grad.setter
    def precond_grad(self,val):
        self._precond_grad = val
    
    @property
    def mask(self):
        return self._mask
    
    @property
    def precondition_scale_factor(self):
        return self._precondition_scale_factor
    
    @property
    def num_parameters(self):
        return self._num_parameters
    
    @property
    def min_bound(self):
        return self._min_bound
    
    @property
    def max_bound(self):
        return self._max_bound
    
    @property
    def theoretic_profile(self):
        return self._theoretic_profile
    
    
class ScattererEstimator(object):
    """
    A ScattererEstimator defines an unknown shdom.Scatterer to be estimated.
    A scatterer estimator contains more basic estimators such as GridDataEstimators which define the parameters of the Scatterer that are to be estimated.
    This is an abstract method that is inherited by a specific type of scatterer estimator (e.g. OpticalScatterEstimator, MicrophysicalScattererEstimator)
    """
    def __init__(self):      
        self._mask = None
        self._estimators = self.init_estimators()
        self._derivatives = self.init_derivatives()
        self._num_parameters = self.init_num_parameters()
        self._num_estimators = len(self.estimators)

    def init_estimators(self):
        """
        Initialize the internal estimators.

        Returns
        -------
        estimators: OrderedDict
            A dictionary of more basic estimators that define the ScattererEstimator.

        Notes
        -----
        This is a dummy method that is overwritten by inheritance.
        """
        return OrderedDict()
    
    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.

        Returns
        -------
        derivatives: OrderedDict
            A dictionary of derivatives that define the ScattererEstimator.

        Notes
        -----
        This is a dummy method that is overwritten by inheritance.
        """
        return OrderedDict()    

    def init_num_parameters(self):
        """
        Initialize the number of parameters to be estimated by accumulating all the internal estimator parameters.

        Returns
        -------
        num_parameters: int
            The number of parameters to be estimated.
        """
        num_parameters = []
        for estimator in self.estimators.values():
            num_parameters.append(estimator.num_parameters)
        return num_parameters

    def set_mask(self, mask):
        """
        Set a mask for data points that will be estimated.
        
        Parameters
        ----------
        mask: shdom.GridData object
            A GridData object with boolean data. True is for data points that will be estimated.
        """
        self._mask = mask.resample(self.grid, method='nearest')
        for estimator in self.estimators.values():
            estimator.set_mask(mask)
        self._num_parameters = self.init_num_parameters()
       
    def set_state(self, state, optimization_iteration = None):
        """
        Set the estimator state by setting all the internal estimators states.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        states = np.split(state, np.cumsum(self.num_parameters[:-1]))
        for estimator, state in zip(self.estimators.values(), states):
            estimator.set_state(state, optimization_iteration)        

    def get_state(self):
        """
        Retrieve the estimator state by joining all the internal estimators states.

        Returns
        -------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        state = np.empty(shape=(0), dtype=np.float64)
        for estimator in self.estimators.values():
            state = np.concatenate((state, estimator.get_state()))
        return state

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter by accumulating from all internal estimators (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        bounds = []
        for estimator in self.estimators.values():
            bounds.extend(estimator.get_bounds())
        return bounds

    def project_gradient(self, gradient, grid):
        """
        Project gradient onto the combined state representation.

        Parameters
        ----------
        grid: shdom.Grid
            The internal shdom grid upon which the gradient was computed.
        gradient: np.array(dtype=np.float64)
            An array containing the gradient of the cost function with respect to the parameters.
        """
        gradient = np.split(gradient, self.num_estimators, axis=-1)
        state_gradient = np.empty(shape=(0), dtype=np.float64)
        debug_gradient_3d = []
        
        for estimator_name, estimator, gradient in zip(self.estimators.keys(), self.estimators.values(), gradient):
            if (estimator_name == 'reff') and ('lwc' in self.estimators.keys()):
                # The lwc will scale the reff averaging in case of 1D reff forcing.
                # The estimator is of type shdom.optimize.GridDataEstimator
                
                state_gradient_new, debug_gradient_3d_new = estimator.project_gradient(gradient, grid, scaling = self.estimators['lwc'].data)
                
            else:
                state_gradient_new, debug_gradient_3d_new = estimator.project_gradient(gradient, grid) # vadim added to get the gradient in the base grid to debug/visualize it later on.

            state_gradient = np.concatenate((state_gradient, state_gradient_new))
            debug_gradient_3d.append(debug_gradient_3d_new)
        return state_gradient, debug_gradient_3d
        
    @property
    def estimators(self):
        return self._estimators

    @property
    def num_estimators(self):
        return self._num_estimators

    @property
    def derivatives(self):
        return self._derivatives    
    
    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def mask(self):
        return self._mask

    
class OpticalScattererEstimator(shdom.OpticalScatterer, ScattererEstimator):
    """
    An OpticalScattererEstimator defines an unknown shdom.OpticalScatterer to be estimated.
    The internal estimators which define an OpticalScatterer are: extinction, albedo, phase.

    Parameters
    ----------
    wavelength: float
        A wavelength in microns
    extinction: shdom.GridData or shdom.GridDataEstimator
        A GridData or GridDataEstimator object containing the extinction (1/km) on a grid
    albedo: shdom.GridData or shdom.GridDataEstimator
        A GridData or GridDataEstimator object containing the single scattering albedo [0,1] on a grid
    phase: shdom.GridPhase or shdom.GridPhaseEstimator
        A GridPhase or GridPhaseEstimator object containing the phase function on a grid

    Notes
    -----
    albedo and phase estimation is not implemented
    """
    def __init__(self, wavelength, extinction, albedo, phase):
        shdom.OpticalScatterer.__init__(self, wavelength, extinction, albedo, phase)
        ScattererEstimator.__init__(self)
        
    def init_estimators(self):
        """
        Initialize the internal estimators: extinction, albedo, phase

        Returns
        -------
        estimators: OrderedDict
            A dictionary with optional GridDataEstimators and/or GridPhaseEstimators

        Notes
        -----
        albedo and phase estimation is not implemented
        """
        estimators = OrderedDict()
        if isinstance(self.extinction, shdom.GridDataEstimator):
            estimators['extinction'] = self.extinction
        if isinstance(self.albedo, shdom.GridDataEstimator):
            raise NotImplementedError("Albedo estimation not implemented")
        if isinstance(self.phase, shdom.GridPhaseEstimator):
            raise NotImplementedError("Phase estimation not implemented")           
        return estimators

    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.
        For this estimator, the internal parameters are the optical fields themselves, hence the derivatives are indicator functions.

        Returns
        -------
        derivatives: OrderedDict of shdom.OpticalScattererDerivative
            A dictionary of derivatives that define the ScattererEstimator.
        """
        derivatives = OrderedDict()
        if isinstance(self.extinction, shdom.GridDataEstimator):
            derivatives['extinction'] = self.init_extinction_derivative()
        if isinstance(self.albedo, shdom.GridDataEstimator):
            derivatives['albedo'] = self.init_albedo_derivative()
        if isinstance(self.phase, shdom.GridPhaseEstimator):
            derivatives['phase'] = self.init_phase_derivative()   
        return derivatives

    def init_extinction_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the optical extinction.
        This is an indicator function, as only the extinction depends on the extinction parameter.

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative object with the optical derivatives with respect to extinction.
        """
        extinction = shdom.GridData(self.extinction.grid, np.ones_like(self.extinction.data))
        albedo = shdom.GridData(self.albedo.grid, np.zeros_like(self.albedo.data))
        if self.phase.legendre_table.table_type == 'SCALAR':
            legcoef = np.zeros((self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        elif self.phase.legendre_table.table_type == 'VECTOR':
            legcoef = np.zeros((self.phase.legendre_table.nstleg, self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        legen_table = shdom.LegendreTable(legcoef, table_type=self.phase.legendre_table.table_type)
        phase = shdom.GridPhase(legen_table, shdom.GridData(self.phase.index.grid, np.ones_like(self.phase.index.data)))
        derivative = shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)        
        return derivative

    def init_albedo_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the single scattering albedo.
        This is an indicator function, as only the albedo depends on the albedo parameter.

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative object with the optical derivatives with respect to albedo.
        """
        extinction = shdom.GridData(self.extinction.grid, np.zeros_like(self.extinction.data))
        albedo = shdom.GridData(self.albedo.grid, np.ones_like(self.albedo.data))
        if self.phase.legendre_table.table_type == 'SCALAR':
            legcoef = np.zeros((self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        elif self.phase.legendre_table.table_type == 'VECTOR':
            legcoef = np.zeros((self.phase.legendre_table.nstleg, self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        legen_table = shdom.LegendreTable(legcoef, table_type=self.phase.legendre_table.table_type)
        phase = shdom.GridPhase(legen_table, shdom.GridData(self.phase.index.grid, np.ones_like(self.phase.index.data)))
        derivative = shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)
        return derivative
    
    def init_phase_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the phase function.

        Notes
        -----
        This is a dummy method which is not implemented.
        """
        raise NotImplementedError("Phase estimation not implemented")  

    def get_derivative(self, derivative_type, wavelength):
        """
        Retrieve the relevant derivative at a single wavelength.

        Parameters
        ----------
        derivative_type: str
            'extinction' for the extinction derivative
            'albedo' for the albedo derivative
            'phase' for the phase derivative
        wavelength: float
            A wavelength in microns

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative with respect to the type requested

        Notes
        -----
        Wavelength here is a dummy. The optical derivatives with respect to the optical parameters are indicator functions and not a function of wavelength
        """
        if derivative_type == 'extinction':
            derivative = self.derivatives['extinction']
        elif derivative_type == 'albedo':
            derivative = self.derivatives['albedo']
        elif derivative_type == 'phase':
            derivative = self.derivatives['phase']   
        else:
            raise AttributeError('derivative type {} not supported'.format(derivative_type))
        return derivative
        
        
class MicrophysicalScattererEstimator(shdom.MicrophysicalScatterer, ScattererEstimator):
    """
    An MicrophysicalScattererEstimator defines an unknown shdom.MicrophysicalScatterer to be estimated.
    The internal estimators which define an MicrophysicalScatterer are: lwc, reff, veff.

    Parameters
    ----------
    mie: shdom.MiePolydisperse or list of shdom.MiePolydisperse
        Using the Mie model microphyical properties are transformed into optical properties (see get_optical_scatterer method)
    lwc: shdom.GridData or shdom.GridDataEstimator
        A GridData object containing liquid water content (g/m^3) on a 3D grid.
    reff: shdom.GridData or shdom.GridDataEstimator
        A GridData object containing effective radii (micron) on a 3D grid.
    veff: shdom.GridDatao r shdom.GridDataEstimator
        A GridData object containing effective variances on a 3D grid.
    """
    def __init__(self, mie, lwc, reff, veff):
        shdom.MicrophysicalScatterer.__init__(self, lwc, reff, veff)
        self.add_mie(mie)
        ScattererEstimator.__init__(self)
        
    def init_estimators(self):
        """
        Initialize the internal estimators: lwc, reff, veff

        Returns
        -------
        estimators: OrderedDict
            A dictionary with optional GridDataEstimators
        """
        estimators = OrderedDict()
        if isinstance(self.lwc, shdom.GridDataEstimator):
            estimators['lwc'] = self.lwc
        if isinstance(self.reff, shdom.GridDataEstimator):
            estimators['reff'] = self.reff
        if isinstance(self.veff, shdom.GridDataEstimator):
            estimators['veff'] = self.veff
        return estimators

    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.
        For this estimator, the internal parameters are: lwc, reff, veff

        Returns
        -------
        derivatives: OrderedDict of shdom.OpticalScattererDerivative
            A dictionary of derivatives that define the ScattererEstimator.
        """
        derivatives = OrderedDict()
        if isinstance(self.lwc, shdom.GridDataEstimator):
            derivatives['lwc'] = self.init_lwc_derivative()
        if isinstance(self.reff, shdom.GridDataEstimator):
            derivatives['reff'] = self.init_mie_derivative(derivative_type='reff')
        if isinstance(self.veff, shdom.GridDataEstimator):
            derivatives['veff'] = self.init_mie_derivative(derivative_type='veff')
        return derivatives

    def init_lwc_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the liquid water content.

        Returns
        -------
        derivative: OrderedDict
            A dictionary of properties that define the the optical derivatives with respect to lwc.

        Notes
        -----
        Only the optical extinction depends on the lwc, however, since it also depends on wavelength it is not initialized here.
        """
        derivative = OrderedDict()
        derivative['lwc'] = shdom.GridData(self.lwc.grid, np.ones_like(self.lwc.data))
        derivative['albedo'] = shdom.GridData(self.grid, np.zeros(self.grid.shape))
        return derivative

    def init_mie_derivative(self, derivative_type):
        """
        Initialize the Mie derivatives of the optical fields with respect to the effective radius/variance.
        This means the optical cross-section, single scattering albedo and phase function derivatives.

        Returns
        -------
        derivatives: OrderedDict
            A dictionary of shdom.MiePolydisperse (at every wavelength).
        derivative_type: str
            The derivative type: 'reff' or 'veff'

        Notes
        -----
        Derivatives are computed numerically thus small enough spacing of reff/veff in the Mie tables is required.
        """
        derivatives = OrderedDict()
        for wavelength, mie in self.mie.items():
            extinct = mie.extinct.reshape((mie.size_distribution.nretab, mie.size_distribution.nvetab), order='F')
            ssalb = mie.ssalb.reshape((mie.size_distribution.nretab, mie.size_distribution.nvetab), order='F')

            if derivative_type == 'reff':
                dextinct = np.gradient(extinct, mie.size_distribution.reff, axis=-2)
                dssalb = np.gradient(ssalb, mie.size_distribution.reff, axis=-2)
                dlegcoef = np.gradient(mie.legcoef_2d, mie.size_distribution.reff, axis=-2)

            elif derivative_type == 'veff':
                dextinct = np.gradient(extinct, mie.size_distribution.veff, axis=-1)
                dssalb = np.gradient(ssalb, mie.size_distribution.veff, axis=-1)
                dlegcoef = np.gradient(mie.legcoef_2d, mie.size_distribution.veff, axis=-1)

            # Define a derivative Mie object, last derivative is duplicated
            derivative = copy.deepcopy(mie)
            derivative._extinct = dextinct.ravel(order='F')
            derivative._ssalb = dssalb.ravel(order='F')
            if mie.table_type == 'SCALAR':
                derivative.legcoef = dlegcoef.reshape((mie.maxleg+1, -1), order='F')
            elif mie.table_type == 'VECTOR':
                derivative.legcoef = dlegcoef.reshape((mie.legendre_table.nstleg, mie.maxleg+1, -1), order='F')
    
            derivative.init_intepolators()
            derivatives[float_round(wavelength)] = derivative
        return derivatives         

    def get_lwc_derivative(self, wavelength):
        """
        Retrieve the liquid water content derivative at a single wavelength.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            The derivative with respect to lwc at a single wavelength
        """
        mie = self.mie[float_round(wavelength)]
        index = shdom.GridData(self.grid, np.ones(self.grid.shape, dtype=np.int32))
        if mie.table_type == 'SCALAR':
            legen_table = shdom.LegendreTable(np.zeros((mie.maxleg+1), dtype=np.float32), mie.table_type)
        elif mie.table_type == 'VECTOR':
            legen_table = shdom.LegendreTable(np.zeros((mie.legendre_table.nstleg, mie.maxleg + 1), dtype=np.float32), mie.table_type)
        derivative = self.derivatives['lwc']
        scatterer = shdom.OpticalScattererDerivative(
            wavelength, 
            extinction=mie.get_extinction(derivative['lwc'], self.reff, self.veff),
            albedo=derivative['albedo'],
            phase=shdom.GridPhase(legen_table, index)) 
        return scatterer

    def get_mie_derivative(self, derivative, wavelength):
        """
        Retrieve mie scattering derivatives at a single wavelength.

        Parameters
        ----------
        derivative: OrderedDict
            A dictionary of shdom.MiePolydisperse at multiple wavelengths
        wavelength: float
            Wavelength in microns. A Mie derivative at this wavelength should be added prior

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            The Mie derivative at a single wavelength
        """
        scatterer = shdom.OpticalScattererDerivative(
            wavelength, 
            extinction=derivative[float_round(wavelength)].get_extinction(self.lwc, self.reff, self.veff),
            albedo=derivative[float_round(wavelength)].get_albedo(self.reff, self.veff),
            phase=derivative[float_round(wavelength)].get_phase(self.reff, self.veff)) 
        return scatterer  

    def get_derivative(self, derivative_type, wavelength):
        """
        Retrieve the relevant derivative at a single wavelength.

        Parameters
        ----------
        derivative_type: str
            'lwc' for the lwc derivative
            'reff' for the reff derivative
            'veff' for the veff derivative
        wavelength: float
            A wavelength in microns

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative with respect to the type requested
        """
        if derivative_type == 'lwc':
            derivative = self.get_lwc_derivative(wavelength)
        elif derivative_type == 'reff' or derivative_type == 'veff':
            derivative = self.get_mie_derivative(self.derivatives[derivative_type], wavelength)
        else:
            raise AttributeError('derivative type {} not supported'.format(derivative_type))
        return derivative


class MediumEstimator(shdom.Medium):
    """
    A MediumEstimator defines an unknown shdom.Medium to be estimated.
    A medium estimator is a shdom.Medium with unknown (and optionally known) scatterers.
    Unknown scatterers are defined by internal estimators (shdom.ScattererEstimator instances).

    Parameters
    ----------
    grid: shdom.Grid, optional
        A grid for the Medium object. All scatterers will be resampled to this grid.
                loss_type: str,
    loss_type: 'l2' or 'normcorr'.
        l2 - used for l2 norm between the acquired and synthetic (rendered) measurements
        normcorr - used for the normalized correlation between the acquired and synthetic (rendered) measurements
    exact_single_scatter: bool
        True will compute the exact single scattering gradient along a broken-ray trajectory (using the direct solar beam)
    stokes_weights: list of floats
        Loss function weights for stokes vector components [I,Q,U,V].
    """
    def __init__(self, grid=None, loss_type='l2', exact_single_scatter=True, stokes_weights=None):
        super().__init__(grid)
        self._estimators = OrderedDict()
        self._num_parameters = []
        self._unknown_scatterers_indices = np.empty(shape=(0), dtype=np.int32)
        self._num_derivatives = 0
        self._num_estimators = 0
        self._exact_single_scatter = exact_single_scatter
        self._core_grad, self._output_transform = self.init_loss_function(loss_type)
        self._stokes_weights = stokes_weights if stokes_weights is not None else np.array([1.0], dtype=np.float32)

    def init_loss_function(self, loss_type):
        """
        Initialized the loss function and corresponding gradient computation.
        This includes how a transformation of the core Fortran output parameters.

        Parameters
        ----------
        loss_type: 'l2' or 'normcorr'.
            l2 - used for l2 norm between the acquired and synthetic (rendered) measurements
            normcorr - used for the normalized correlation between the acquired and synthetic (rendered) measurements

        Returns
        -------
        core_grad: function
             The core gradient and loss computation routine
        output_transform: function
             The transformation to the output parameters.
        """
        if loss_type == 'l2':
            core_grad = self.grad_l2

            def output_transform(output, projection, sensor, num_wavelengths):
                loss = np.sum(list(map(lambda x: x[1], output)))
                gradient = np.sum(list(map(lambda x: x[0], output)), axis=0)
                images = sensor.make_images(np.concatenate(list(map(lambda x: x[2], output)), axis=-1),
                                            projection,
                                            num_wavelengths)
                return loss, gradient, images

        elif loss_type == 'normcorr':
            core_grad = self.grad_normcorr

            def output_transform(output, projection, sensor, num_wavelengths):
                norm1 = np.sum(list(map(lambda x: x[2], output)), axis=0)[:, None, None]
                norm2 = np.sum(list(map(lambda x: x[3], output)), axis=0)[:, None, None]
                norm = np.sqrt(norm1 * norm2)
                loss = np.sum(list(map(lambda x: x[4], output)), axis=0)[:, None, None]
                grad_fn = lambda x: ((loss * x[0]) / norm1 - x[1]) / norm
                gradient = np.mean(np.sum(list(map(grad_fn, output)), axis=0), axis=0)
                loss = -np.mean(loss / norm, dtype=np.float64)
                images = sensor.make_images(np.concatenate(list(map(lambda x: x[5], output)), axis=-1),
                                            projection,
                                            num_wavelengths)
                return loss, gradient, images
        else:
            raise NotImplementedError('Loss type {} not implemented'.format(loss_type))

        return core_grad, output_transform

    def add_scatterer(self, scatterer, name=None):
        """
        Add a Scatterer to the medium. If the scatterer is a ScattererEstimator it will enter the estimator list.

        Parameters
        ----------
        scatterer: shdom.Scatterer or shdom.ScattererEstimator
            A known or unknown scattering particle distribution
            (e.g. MicrophysicalScatterer, MicrophysicalScattererEstimator, OpticalScatterer, OpticalScattererEstimator)
        name: string, optional
            A name for the scatterer that will be used to retrieve it (see get_scatterer method).
            If no name is specified the default name is scatterer# where # is the number in which it was input (i.e. scatterer1 for the first scatterer).
        """
        super().add_scatterer(scatterer, name)
        if issubclass(type(scatterer), shdom.ScattererEstimator):
            name = 'scatterer{:d}'.format(self._num_scatterers) if name is None else name
            self._num_estimators += 1
            total_num_estimators = len(scatterer.estimators)
            self._estimators[name] = scatterer
            self._num_parameters.append(np.sum(scatterer.num_parameters))
            self._unknown_scatterers_indices = np.concatenate((
                self.unknown_scatterers_indices, 
                np.full(total_num_estimators, self.num_scatterers, dtype=np.int32)))
            self._num_derivatives += total_num_estimators

    def set_state(self, state, optimization_iteration = None):
        """
        Set the estimator state by setting all the internal estimators states.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        states = np.split(state, np.cumsum(self.num_parameters[:-1]))
        for (name, estimator), state in zip(self.estimators.items(), states):
            estimator.set_state(state, optimization_iteration)
            self.scatterers[name] = estimator

    def get_state(self):
        """
        Retrieve the estimator state by joining all the internal estimators states.

        Returns
        -------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        state = np.empty(shape=(0),dtype=np.float64)
        for estimator in self.estimators.values():
            state = np.concatenate((state, estimator.get_state()))
        return state

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter by accumulating from all internal estimators (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        bounds = []
        for estimator in self.estimators.values():
            bounds.extend(estimator.get_bounds())
        return bounds

    def get_derivatives(self, rte_solver):
        """
        Retrieve the relevant derivatives for a given RteSolver.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver object (at a given wavelength)

        Returns
        -------
        dext: np.array(dtype=np.float32)
            The derivative of the optical extinction with respect to the parameters
        dalb: np.array(dtype=np.float32)
            The derivative of the single scattering albedo with respect to the parameters
        diphase: np.array(dtype=np.int32)
            A pointer to the derivative of the phase function respect to the parameters
        dleg: np.array(dtype=np.float32)
             The derivative of the phase function legendre coefficients with respect to the parameters
        dphasetab: np.array(dtype=np.float32)
             The derivative of the phase function at pre-determined angles with respect to the parameters
        dnumphase: np.array(dtype=np.float32)
             The number of phase function derivatives
        """
        dext = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.float32)
        dalb = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.float32)
        diphase = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.int32)
    
        i=0
        for estimator in self.estimators.values():
            for dtype in estimator.derivatives.keys():
                derivative = estimator.get_derivative(dtype, rte_solver.wavelength)       
                resampled_derivative = derivative.resample(self.grid)
                dext[:, i] = resampled_derivative.extinction.data.ravel()
                dalb[:, i] = resampled_derivative.albedo.data.ravel()
                diphase[:, i] = resampled_derivative.phase.iphasep.ravel() + diphase.max()

                if i == 0:
                    leg_table = copy.deepcopy(resampled_derivative.phase.legendre_table)
                else:
                    leg_table.append(copy.deepcopy(resampled_derivative.phase.legendre_table))                
                i += 1
                
        leg_table.pad(rte_solver._nleg)
        dleg = leg_table.data
        dnumphase = leg_table.numphase
        
        # zero the first term of the first component of the phase function
        # gradient. Pre-scale the legendre moments by 1/(2*l+1) which
        # is done in the forward problem in TRILIN_INTERP_PROP
        scaling_factor =np.array([2.0*i+1.0 for i in range(0,rte_solver._nleg+1)])
        if dleg.ndim == 2:
            dleg[0,:] = 0.0
            dleg /= scaling_factor[:,np.newaxis]
        elif dleg.ndim ==3:
            dleg[0,0,:] = 0.0
            dleg /= scaling_factor[np.newaxis,:,np.newaxis]

        dphasetab = core.precompute_phase_check_grad(
            negcheck=False,
            nstphase=rte_solver._nstphase,
            nstleg=rte_solver._nstleg,
            nscatangle=rte_solver._nscatangle,
            nstokes=rte_solver._nstokes,
            dnumphase=dnumphase,
            ml=rte_solver._ml,
            nlm=rte_solver._nlm,
            nleg=rte_solver._nleg,
            dleg=dleg,
            deltam=rte_solver._deltam
        )

        return dext, dalb, diphase, dleg, dphasetab, dnumphase

    def compute_direct_derivative(self, rte_solver):
        """
        Compute the derivative with respect to the direct solar beam. This is a ray for every point in 3D space.
        Internally this method stores for every point the indices and paths traversed by the direct solar beam to reach it.
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver object (at a given wavelength)
        """

        # There is no optical information stored here, only paths and indices
        # Therefor, there is no important for the specific RteSolver which is used
        if isinstance(rte_solver, shdom.RteSolverArray):
            uniformzlev = max([solver._uniformzlev for solver in rte_solver])
            rte_solver = rte_solver[0]
        else:
            uniformzlev = rte_solver._uniformzlev
            
        self._direct_derivative_path, self._direct_derivative_ptr = \
            core.make_direct_derivative(
                npts=rte_solver._npts,
                bcflag=rte_solver._bcflag,
                gridpos=rte_solver._gridpos,
                npx=rte_solver._pa.npx,
                npy=rte_solver._pa.npy,
                npz=rte_solver._pa.npz,
                delx=rte_solver._pa.delx,
                dely=rte_solver._pa.dely,
                xstart=rte_solver._pa.xstart,
                ystart=rte_solver._pa.ystart,
                zlevels=rte_solver._pa.zlevels,
                ipdirect=rte_solver._ipdirect,
                di=rte_solver._di,
                dj=rte_solver._dj,
                dk=rte_solver._dk,
                epss=rte_solver._epss,
                epsz=rte_solver._epsz,
                xdomain=rte_solver._xdomain,
                ydomain=rte_solver._ydomain,
                cx=rte_solver._cx,
                cy=rte_solver._cy,
                cz=rte_solver._cz,
                cxinv=rte_solver._cxinv,
                cyinv=rte_solver._cyinv,
                czinv=rte_solver._czinv,
                uniformzlev=uniformzlev,
                delxd=rte_solver._delxd,
                delyd=rte_solver._delyd
            )       
        
    def grad_normcorr(self, rte_solver, projection, pixels):
        """
        The core normalized correlation gradient method.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            A solver with all the associated parameters and the solution to the RTE
        projection: shdom.Projection
            A projection model which specified the position and direction of each and every pixel
        pixels: np.array(shape=(projection.npix), dtype=np.float32)
            The acquired pixels driving the error and optimization.

        Returns
        -------
        grad1: np.array(shape=(rte_solver._nstokes, rte_solver._nbpts, self.num_derivatives), dtype=np.float32)
            A part of the gradient with respect to all parameters at every grid base point
        grad2: np.array(shape=(rte_solver._nstokes, rte_solver._nbpts, self.num_derivatives), dtype=np.float32)
            A part of the gradient with respect to all parameters at every grid base point
        norm1: np.array(shape=(rte_solver._nstokes), dtype=np.float32)
            The per stokes component norm of all synthetic pixels
        norm2: np.array(shape=(rte_solver._nstokes), dtype=np.float32)
            The per stokes component norm of all measurements
        loss: float32
            The per stokes component correlations of the measruements and synthetic pixels
        images: np.array(shape=(rte_solver._nstokes, projection.npix), dtype=np.float32)
            The rendered (synthetic) images.
        """
        if isinstance(projection.npix, list):
            total_pix = np.sum(projection.npix)
        else:
            total_pix = projection.npix

        grad1, grad2, norm1, norm2, loss, images = core.gradient_normcorr(
            weights=self._stokes_weights[:rte_solver._nstokes],
            exact_single_scatter=self._exact_single_scatter,
            nstphase=rte_solver._nstphase,
            dpath=self._direct_derivative_path, 
            dptr=self._direct_derivative_ptr,
            npx=rte_solver._pa.npx,
            npy=rte_solver._pa.npy,
            npz=rte_solver._pa.npz,
            delx=rte_solver._pa.delx,
            dely=rte_solver._pa.dely,                
            xstart=rte_solver._pa.xstart,
            ystart=rte_solver._pa.ystart,
            zlevels=rte_solver._pa.zlevels, 
            extdirp=rte_solver._pa.extdirp,
            uniformzlev=rte_solver._uniformzlev,
            partder=self.unknown_scatterers_indices,
            numder=self.num_derivatives,
            dext=rte_solver._dext,
            dalb=rte_solver._dalb,
            diphase=rte_solver._diphase,
            dleg=rte_solver._dleg,
            dphasetab=rte_solver._dphasetab,
            dnumphase=rte_solver._dnumphase,
            nscatangle=rte_solver._nscatangle,
            phasetab=rte_solver._phasetab,
            ylmsun=rte_solver._ylmsun,
            nstokes=rte_solver._nstokes,
            nstleg=rte_solver._nstleg,
            nx=rte_solver._nx,
            ny=rte_solver._ny,
            nz=rte_solver._nz,
            bcflag=rte_solver._bcflag,
            ipflag=rte_solver._ipflag,   
            npts=rte_solver._npts,
            nbpts=rte_solver._nbpts,
            ncells=rte_solver._ncells,
            nbcells=rte_solver._nbcells,
            ml=rte_solver._ml,
            mm=rte_solver._mm,
            ncs=rte_solver._ncs,
            nlm=rte_solver._nlm,
            numphase=rte_solver._pa.numphase,
            nmu=rte_solver._nmu,
            nphi0max=rte_solver._nphi0max,
            nphi0=rte_solver._nphi0,
            maxnbc=rte_solver._maxnbc,
            ntoppts=rte_solver._ntoppts,
            nbotpts=rte_solver._nbotpts,
            nsfcpar=rte_solver._nsfcpar,
            gridptr=rte_solver._gridptr,
            neighptr=rte_solver._neighptr,
            treeptr=rte_solver._treeptr,             
            shptr=rte_solver._shptr,
            bcptr=rte_solver._bcptr,
            cellflags=rte_solver._cellflags,
            iphase=rte_solver._iphase[:rte_solver._npts],
            deltam=rte_solver._deltam,
            solarflux=rte_solver._solarflux,
            solarmu=rte_solver._solarmu,
            solaraz=rte_solver._solaraz,
            gndtemp=rte_solver._gndtemp,
            gndalbedo=rte_solver._gndalbedo,
            skyrad=rte_solver._skyrad,
            waveno=rte_solver._waveno,
            wavelen=rte_solver._wavelen,
            mu=rte_solver._mu,
            phi=rte_solver._phi,
            wtdo=rte_solver._wtdo,
            xgrid=rte_solver._xgrid,
            ygrid=rte_solver._ygrid,
            zgrid=rte_solver._zgrid,
            gridpos=rte_solver._gridpos,
            sfcgridparms=rte_solver._sfcgridparms,
            bcrad=rte_solver._bcrad,
            extinct=rte_solver._extinct[:rte_solver._npts],
            albedo=rte_solver._albedo[:rte_solver._npts],
            legen=rte_solver._legen,
            dirflux=rte_solver._dirflux[:rte_solver._npts],
            fluxes=rte_solver._fluxes,
            source=rte_solver._source,
            camx=projection.x,
            camy=projection.y,
            camz=projection.z,
            cammu=projection.mu,
            camphi=projection.phi,
            npix=total_pix,       
            srctype=rte_solver._srctype,
            sfctype=rte_solver._sfctype,
            units=rte_solver._units,
            measurements=pixels,
            rshptr=rte_solver._rshptr,
            radiance=rte_solver._radiance,
            total_ext=rte_solver._total_ext[:rte_solver._npts]
        )
        return grad1, grad2, norm1, norm2, loss, images

    def grad_l2(self, rte_solver, projection, pixels):
        """
        The core l2 gradient method.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            A solver with all the associated parameters and the solution to the RTE
        projection: shdom.Projection
            A projection model which specified the position and direction of each and every pixel
        pixels: np.array(shape=(projection.npix), dtype=np.float32)
            The acquired pixels driving the error and optimization.

        Returns
        -------
        gradient: np.array(shape=(rte_solver._nbpts, self.num_derivatives), dtype=np.float64)
            The gradient with respect to all parameters at every grid base point
        loss: float64
            The total loss accumulated over all pixels
        images: np.array(shape=(rte_solver._nstokes, projection.npix), dtype=np.float32)
            The rendered (synthetic) images.
        """
        c = 0
        if isinstance(projection.npix, list):
            total_pixs = np.sum(projection.npix)
            rays_per_pixel = []
            for samples_per_pixel, npix in zip(projection.samples_per_pixel,projection.npix):
                rays = np.repeat(samples_per_pixel, npix)
                rays_per_pixel.append(rays)  
            rays_per_pixel = np.concatenate(rays_per_pixel)
        else:
            total_pixs = projection.npix
            rays_per_pixel = np.repeat(projection.samples_per_pixel, total_pixs)
            c = c + 1
    
        rays_weights = projection.weight # pixel/ray weight of contibution to gradient due to samples per pixel introdution, interduced by vadim.
        grad_geo_weights = projection.additional_weight

        #print('--> rays per pixel {}\n, total_pixs {}'.format(rays_per_pixel, total_pixs))

        gradient, loss, images = core.gradient_l2(
            weights=self._stokes_weights[:rte_solver._nstokes],
            exact_single_scatter=self._exact_single_scatter,
            nstphase=rte_solver._nstphase,
            dpath=self._direct_derivative_path,
            dptr=self._direct_derivative_ptr,
            npx=rte_solver._pa.npx,
            npy=rte_solver._pa.npy,
            npz=rte_solver._pa.npz,
            delx=rte_solver._pa.delx,
            dely=rte_solver._pa.dely,
            xstart=rte_solver._pa.xstart,
            ystart=rte_solver._pa.ystart,
            zlevels=rte_solver._pa.zlevels,
            extdirp=rte_solver._pa.extdirp,
            uniformzlev=rte_solver._uniformzlev,
            partder=self.unknown_scatterers_indices,
            numder=self.num_derivatives,
            dext=rte_solver._dext,
            dalb=rte_solver._dalb,
            diphase=rte_solver._diphase,
            dleg=rte_solver._dleg,
            dphasetab=rte_solver._dphasetab,
            dnumphase=rte_solver._dnumphase,
            nscatangle=rte_solver._nscatangle,
            phasetab=rte_solver._phasetab,
            ylmsun=rte_solver._ylmsun,
            nstokes=rte_solver._nstokes,
            nstleg=rte_solver._nstleg,
            nx=rte_solver._nx,
            ny=rte_solver._ny,
            nz=rte_solver._nz,
            bcflag=rte_solver._bcflag,
            ipflag=rte_solver._ipflag,
            npts=rte_solver._npts,
            nbpts=rte_solver._nbpts,
            ncells=rte_solver._ncells,
            nbcells=rte_solver._nbcells,
            ml=rte_solver._ml,
            mm=rte_solver._mm,
            ncs=rte_solver._ncs,
            nlm=rte_solver._nlm,
            numphase=rte_solver._pa.numphase,
            nmu=rte_solver._nmu,
            nphi0max=rte_solver._nphi0max,
            nphi0=rte_solver._nphi0,
            maxnbc=rte_solver._maxnbc,
            ntoppts=rte_solver._ntoppts,
            nbotpts=rte_solver._nbotpts,
            nsfcpar=rte_solver._nsfcpar,
            gridptr=rte_solver._gridptr,
            neighptr=rte_solver._neighptr,
            treeptr=rte_solver._treeptr,
            shptr=rte_solver._shptr,
            bcptr=rte_solver._bcptr,
            cellflags=rte_solver._cellflags,
            iphase=rte_solver._iphase[:rte_solver._npts],
            deltam=rte_solver._deltam,
            solarflux=rte_solver._solarflux,
            solarmu=rte_solver._solarmu,
            solaraz=rte_solver._solaraz,
            gndtemp=rte_solver._gndtemp,
            gndalbedo=rte_solver._gndalbedo,
            skyrad=rte_solver._skyrad,
            waveno=rte_solver._waveno,
            wavelen=rte_solver._wavelen,
            mu=rte_solver._mu,
            phi=rte_solver._phi,
            wtdo=rte_solver._wtdo,
            xgrid=rte_solver._xgrid,
            ygrid=rte_solver._ygrid,
            zgrid=rte_solver._zgrid,
            gridpos=rte_solver._gridpos,
            sfcgridparms=rte_solver._sfcgridparms,
            bcrad=rte_solver._bcrad,
            extinct=rte_solver._extinct[:rte_solver._npts],
            albedo=rte_solver._albedo[:rte_solver._npts],
            legen=rte_solver._legen,
            dirflux=rte_solver._dirflux[:rte_solver._npts],
            fluxes=rte_solver._fluxes,
            source=rte_solver._source,
            camx=projection.x,
            camy=projection.y,
            camz=projection.z,
            ray_weights=rays_weights, 
            rays_per_pixel = rays_per_pixel,
            grad_geo_weights = grad_geo_weights,# interduced by vadim to scale the gradient differently per view.
            cammu=projection.mu,
            camphi=projection.phi,
            npix=total_pixs,
            srctype=rte_solver._srctype,
            sfctype=rte_solver._sfctype,
            units=rte_solver._units,
            measurements=pixels,
            rshptr=rte_solver._rshptr,
            radiance=rte_solver._radiance,
            total_ext=rte_solver._total_ext[:rte_solver._npts]
        )
        #print("-------")
        #print('{} - total pixels is {}'.format(c,total_pixs))        
        #print('{} - loss is {}'.format(c,loss))
        return gradient, loss, images

    def compute_cost(self, rte_solvers, measurements, n_jobs):
        """
        Vadim added to compute only the cost of the current state.
        If n_jobs>1 than parallel gradient computation is used with pixels are distributed amongst all workers
        
        Parameters
        ----------
        rte_solvers: shdom.RteSolverArray
            A solver array with all the associated parameters and the solution to the RTE
        measurements: shdom.Measurements or shdom.CloudCT_setup.SpaceMultiView_Measurements
            A measurements object storing the acquired images and sensor geometry
        n_jobs: int,
            The number of jobs to divide the gradient computation into.

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels.
        """   
        loss = 0
        if(isinstance(measurements,shdom.CloudCT_setup.SpaceMultiView_Measurements)):
            # relating to CloudCT multi-view setup with different imagers.
            CloudCT_geometry_and_imagers = measurements.setup
            images_dict = measurements.images.copy() # if the measurments of type cloudct...., the images in grayscale.
            
            imagers_channels = measurements.get_channels_of_imagers() # A list, it is the central wavelength of al imagers. 
            for imager_index, wavelength in enumerate(imagers_channels):
                acquired_images = images_dict[imager_index]
                CloudCT_projection = CloudCT_geometry_and_imagers[imager_index]
                
                if(measurements.sensor_type == 'RadianceSensor'):
                    sensor=shdom.RadianceSensor()  
                    
                elif(measurements.sensor_type == 'StokesSensor'):
                    sensor=shdom.StokesSensor()
                else:
                    raise Exception('Unsupported')
                
                camera = shdom.Camera(sensor, CloudCT_projection)
                images_per_imager = camera.render(rte_solvers[imager_index], n_jobs=n_jobs)  
                                  
                for (i_measured,i) in zip(acquired_images,images_per_imager):
                    loss += 0.5*np.sum(np.power((i_measured.ravel() - i.ravel()),2))
                
                            
                            
        else:
            raise Exception('Unsupported')
            #projection = measurements.camera.projection
            #sensor = measurements.camera.sensor
            #pixels = measurements.pixels            
         
        return loss   
        
    def compute_gradient(self, rte_solvers, measurements, n_jobs, iteration):
        """
        Compute the gradient with respect to the current state.
        If n_jobs>1 than parallel gradient computation is used with pixels are distributed amongst all workers
        
        Parameters
        ----------
        rte_solvers: shdom.RteSolverArray
            A solver array with all the associated parameters and the solution to the RTE
        measurements: shdom.Measurements or shdom.CloudCT_setup.SpaceMultiView_Measurements
            A measurements object storing the acquired images and sensor geometry
        n_jobs: int,
            The number of jobs to divide the gradient computation into.

        Returns
        -------
        state_gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the loss function with respect to the state parameters
        loss: np.float64
            The total loss accumulated over all pixels
        images: list of np.array(shape=(measurements.projection.resolution), dtype=np.float32)
            A list of the rendered (synthetic) images, used for display purposes.
        """
        # Pre-computation of phase-function and derivatives for all solvers.
        for rte_solver in rte_solvers.solver_list:
            rte_solver.precompute_phase()
            rte_solver._dext, rte_solver._dalb, rte_solver._diphase, \
                rte_solver._dleg, rte_solver._dphasetab, rte_solver._dnumphase = self.get_derivatives(rte_solver)


        IF_SHOW_DERIVATIVES = False
        if IF_SHOW_DERIVATIVES:
            
            
            names = ['dext', 'dalb', 'diphase']
            
            for rte_solver in rte_solvers.solver_list:    
                
                DERIVATIVES = [rte_solver._dext, rte_solver._dalb, rte_solver._diphase]
                
                wavelength_nm = int(1e3*rte_solver.wavelength)
                for estimator  in self.estimators.values():
                    for DERIVATIVE,name  in zip(DERIVATIVES, names):
                        DERIVATIVE = DERIVATIVE.reshape(self.grid.shape + tuple([self.num_derivatives]))
                        _ , DERIVATIVE_3d = estimator.project_gradient(DERIVATIVE, self.grid) #  
                        
                        
                        filename = 'lwc_{}_at_{}nm_iter_{}.mat'.format(name,wavelength_nm,iteration)                                        
                        sio.savemat(filename, {'data':DERIVATIVE_3d[0]})                    
                                  
                        filename = 'reff_{}_at_{}nm_iter_{}.mat'.format(name,wavelength_nm,iteration)                                        
                        sio.savemat(filename, {'data':DERIVATIVE_3d[1]}) 
                print('-----------------------')
                    
        if(isinstance(measurements,shdom.CloudCT_setup.SpaceMultiView_Measurements)):
            # relating to CloudCT multi-view setup with different imagers.
            CloudCT_geometry_and_imagers = measurements.setup
            images_dict = measurements.images.copy() # if the measurments of type cloudct...., the images in grayscale.
            images = [] # will gather the output of the gradient calculation.
            gradient = np.zeros(shape=(rte_solver._nbpts, self.num_derivatives), dtype=np.float64)
            loss = 0
            
            imagers_channels = measurements.get_channels_of_imagers() # A list, it is the central wavelength of al imagers. 
            for imager_index, wavelength in enumerate(imagers_channels):
                acquired_images = images_dict[imager_index]
                CloudCT_projection = CloudCT_geometry_and_imagers[imager_index]
                # resample rays per pixels to imitate random sampling of the rays per pixel:
                #if any(np.array(CloudCT_projection.samples_per_pixel)>1):
                    #CloudCT_projection.resample_rays_per_pixel()
                    
                # chack consistensy in the wavelengths:
                assert wavelength == CloudCT_projection.imager.centeral_wavelength_in_microns,\
                       "There is not consistency between the wavelengths between the imager and the CLoudCT setup measurements."
                if(isinstance(rte_solvers.wavelength,list)):
                    
                    assert wavelength == rte_solvers.wavelength[imager_index],\
                           "There is not consistency between the wavelengths between the rte solver and the CLoudCT setup measurements."
                else:
                    assert wavelength == rte_solvers.wavelength,\
                           "There is not consistency between the wavelengths between the rte solver and the CLoudCT setup measurements."
                    
                if(measurements.sensor_type == 'RadianceSensor'):
                    sensor=shdom.RadianceSensor()
                    num_channels = CloudCT_projection.num_channels
                    pixels = []
                    rays_per_pixel = []
                    for (image, samples_per_pixel) in zip(acquired_images,CloudCT_projection.samples_per_pixel):
                        # duplicate pixels:
                        # row_pixels = np.repeat(image.reshape((-1, num_channels), order='F'), samples_per_pixel)[:,np.newaxis]
                        # don't replicate pixels, leave them as they are:
                        row_pixels = image.reshape((-1, num_channels), order='F')
                        pixels.append(row_pixels)
                        rays = np.repeat(samples_per_pixel, row_pixels.size)
                        rays_per_pixel.append(rays)
                        
                    pixels = np.concatenate(pixels, axis=-2)
                    rays_per_pixel = np.concatenate(rays_per_pixel)
                    
                else:
                    raise AttributeError('Only RadianceSensor is supported with the CloudCT setups.')                
                
                # ------ I AM HERE IN THE REIMPLEMENTATION -----
                # TODO- consider to use num_channels>1 per one imager.
                # Sequential or parallel processing using multithreading (threadsafe Fortran)
                #n_jobs = 1
                
                if n_jobs > 1:          
                    
                    # special split for n_jobs != N_views:
                    n_jobs_projections = CloudCT_projection.split(n_jobs)
                    special_split = [p.npix for p in n_jobs_projections]
                    test_split = [int(p.nrays/p.samples_per_pixel) for p in n_jobs_projections]
                    assert special_split == test_split , 'wrong rays counting or spliting'
                    n_jobs_split_pixels = np.split(pixels, np.cumsum(special_split[:-1]))
                    
                    output = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                        delayed(self.core_grad, check_pickle=False)(
                            rte_solver=rte_solvers[loop_imager_index],
                            projection=projection,
                            pixels=loop_pixels
                        ) for loop_imager_index, projection, loop_pixels in
                        zip(np.tile(imager_index,n_jobs) , n_jobs_projections, n_jobs_split_pixels )
                    )
                    
                    #for loop_imager_index, projection, loop_pixels in zip(np.tile(imager_index,n_jobs, CloudCT_projection.split(n_jobs), np.array_split(pixels, n_jobs, axis=-2)):
                        #print(loop_imager_index)
                    
                else:
                    output = [self.core_grad(rte_solvers[imager_index], CloudCT_projection, pixels)]


                """ 
                What we should get in the output?
                gradient: np.array(shape=(rte_solver._nbpts, self.num_derivatives), dtype=np.float64)
                    The gradient with respect to all parameters at every grid base point.
                    The number of grid base points is rte_solver._nbpts = nx*ny*nz of the whole medium (includig Rayleigh).
                loss: float64
                    The total loss accumulated over all pixels
                images: np.array(shape=(rte_solver._nstokes, projection.npix), dtype=np.float32)
                    The rendered (synthetic) images.
                      
                Current problems:
                1. The PIXEL_ERROR = VISOUT(N) - MEASUREMENTS(N) expresions in the SUBROUTINE GRADIENT_L2
                in shdomsub4.f. We must scale the VISOUT with the radiance to grayscale converstion scaleing.
                """ 
                #loss_per_imager = output[1]
                loss_per_imager = np.sum(list(map(lambda x: x[1], output)))
                gradient_per_njob = list(map(lambda x: x[0], output))
                #print(CloudCT_projection.view_geometric_scaling)
                #gradient_per_njob = [ i*g for i,g in zip(CloudCT_projection.view_geometric_scaling,gradient_per_njob)]
                gradient_per_imager = np.sum(gradient_per_njob, axis=0)    
                images_per_imager = sensor.inverse_make_images(np.concatenate(list(map(lambda x: x[2], output)), axis=-1),
                                            CloudCT_projection,
                                            num_channels)   
                
                #if any(np.array(CloudCT_projection.samples_per_pixel)>1):
                                    #CloudCT_projection.resample_rays_per_pixel()
                                    ## If any projection has samples_per_pixel > 1, the cost calculated inside GRADIENT_L2 (fortran) is wrong.
                                    ## It is refined here:
                                    #loss_per_imager = 0
                                    #for (i_measured,i) in zip(acquired_images,images_per_imager):
                                        #loss_per_imager += 0.5*np.sum(np.power((i_measured.ravel() - i.ravel()),2))
                
                #if(imager_index == 1):
                    #gradient_per_imager = 5*gradient_per_imager
                    
                gradient = np.sum([gradient, gradient_per_imager], axis=0) 
                loss += loss_per_imager
                images += images_per_imager
             
            # save imaged for the debugging.    
            if(iteration == -1):
                measurment_images = measurements.images
                counter = 0
                pad = 0.01
                from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
              
                for imager_index in measurment_images.keys():
                    
                    for view_index, img in enumerate(measurment_images[imager_index]):
                        name = view_index
                        fig, ax = plt.subplots(1, 3, figsize=(20, 20))
                        img2 = images[counter]
                        img1 = img
                        MAXI = max(np.max(img1), np.max(img2))
                        MINI = min(np.min(img1), np.min(img2))
                        im1 = ax[0].imshow(img1,cmap='jet',vmin=MINI, vmax=MAXI)
                        ax[0].set_title("{}_1".format(name))
                        divider = make_axes_locatable(ax[0])
                        cax = divider.append_axes("right", size="5%", pad=pad)
                        plt.colorbar(im1, cax=cax)        
                        
                        im2 = ax[1].imshow(img2,cmap='jet',vmin=MINI, vmax=MAXI)
                        ax[1].set_title("{}_2".format(name))
                        divider = make_axes_locatable(ax[1])
                        cax = divider.append_axes("right", size="5%", pad=pad)
                        plt.colorbar(im2, cax=cax)        
                        
                        im3 = ax[2].imshow(img1-img2,cmap='jet')
                        ax[2].set_title("diff.")
                        divider = make_axes_locatable(ax[2])
                        cax = divider.append_axes("right", size="5%", pad=pad)
                        plt.colorbar(im3, cax=cax) 
                        
                        counter += 1
                plt.show()
            #-----------------------------------
            
                #import scipy.io as sio
                #file_name = '133_state_images.mat'
                #sio.savemat(file_name, {'images':images})
                #print('---images of iteration {} were saved.-----'.format(iteration))                
                #see images of a state:
                #fig, ax = plt.subplots(2, 5, figsize=(20, 20))
                #from mpl_toolkits.axes_grid1 import make_axes_locatable
                
                #ax = ax.flatten()
                #MAXI = np.array(images_per_imager).max()
                #for index, img in enumerate(images_per_imager):
                    #im = ax[index].imshow(img,cmap='gray',vmin=0, vmax=MAXI)
                    #ax[index].set_title("{}".format(index))
                    #divider = make_axes_locatable(ax[index])
                    #cax = divider.append_axes("right", size="5%", pad=0.01)
                    #plt.colorbar(im, cax=cax)                    
                #plt.show()                
                
            # Outside the imager_index loop:
            gradient = gradient.reshape(self.grid.shape + tuple([self.num_derivatives])) # An array containing the gradient of the cost function with respect to the parameters. the shape is of the internal shdom grid upon which the gradient was computed.
            gradient = np.split(gradient, self.num_estimators, axis=-1)# gradient is now a list with num_estimators elements. 
            state_gradient = np.empty(shape=(0), dtype=np.float64) # State gradient representation
            for estimator, gradient in zip(self.estimators.values(), gradient):
                estimator_project_gradient, debug_gradient_3d = estimator.project_gradient(gradient, self.grid) # vadim added to get the gradient in the base grid to debug/visualize it later on. 
                state_gradient = np.concatenate((state_gradient, estimator_project_gradient))
                # Note that the debug_gradient_3d will be saved only for 1 estimator.                
            
        else:
            projection = measurements.camera.projection
            sensor = measurements.camera.sensor
            pixels = measurements.pixels
        
            # Sequential or parallel processing using multithreading (threadsafe Fortran)
            if n_jobs > 1:           
                output = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                    delayed(self.core_grad, check_pickle=False)(
                        rte_solver=rte_solvers[channel],
                        projection=projection,
                        pixels=spectral_pixels[..., channel]
                    ) for channel, (projection, spectral_pixels) in
                    itertools.product(range(self.num_wavelengths), zip(projection.split(n_jobs), np.array_split(pixels, n_jobs, axis=-2)))
                )
            else:
                output = [
                    self.core_grad(rte_solvers[channel], projection, pixels[..., channel])
                    for channel in range(self.num_wavelengths)
                ]
            
            # Sum over all the losses of the different channels
            loss, gradient, images = self.output_transform(output, projection, sensor, self.num_wavelengths)
    
            gradient = gradient.reshape(self.grid.shape + tuple([self.num_derivatives]))
            gradient = np.split(gradient, self.num_estimators, axis=-1)
            state_gradient = np.empty(shape=(0), dtype=np.float64)
            for estimator, gradient in zip(self.estimators.values(), gradient):
                estimator_project_gradient, debug_gradient_3d = estimator.project_gradient(gradient, self.grid) # vadim added to get the gradient in the base grid to debug/visualize it later on. 
                state_gradient = np.concatenate((state_gradient, estimator_project_gradient))
                # Note that the debug_gradient_3d will be saved only for 1 estimator.                

        return state_gradient, loss, images, debug_gradient_3d

    @property
    def estimators(self):
        return self._estimators

    @property
    def num_parameters(self):
        return self._num_parameters  

    @property
    def unknown_scatterers_indices(self):
        return self._unknown_scatterers_indices

    @property
    def num_derivatives(self):
        return self._num_derivatives 

    @property
    def num_estimators(self):
        return self._num_estimators

    @property
    def core_grad(self):
        return self._core_grad

    @property
    def output_transform(self):
        return self._output_transform


class SummaryWriter(object):
    """
    A wrapper for tensorboardX summarywriter with some basic summary writing implementation.
    This wrapper enables logging of images, error measures and loss with pre-determined temporal intervals into tensorboard.

    To view the summary of this run (and comparisons to all subdirectories):
        tensorboard --logdir LOGDIR 

    Parameters
    ----------
    log_dir: str
        The directory where the log will be saved
    """
    def __init__(self, log_dir=None):
        self._dir = log_dir
        self._tf_writer = tb.SummaryWriter(log_dir) if log_dir is not None else None
        self._ground_truth_parameters = None
        self._callback_fns = []
        self._kwargs = []
        self._optimizer = None

    def add_callback_fn(self, callback_fn, kwargs=None):
        """
        Add a callback function to the callback function list

        Parameters
        ----------
        callback_fn: bound method
            A callback function to push into the list
        kwargs: dict, optional
            A dictionary with optional keyword arguments for the callback function
        """
        self._callback_fns.append(callback_fn)
        self._kwargs.append(kwargs)

    def attach_optimizer(self, optimizer):
        """
        Attach the optimizer

        Parameters
        ----------
        optimizer: shdom.Optimizer
            The optimizer that the writer will report for
        """
        self._optimizer = optimizer

    def monitor_loss(self, ckpt_period=-1):
        """
        Monitor the loss.
        
        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'loss'
        }
        self.add_callback_fn(self.loss_cbfn, kwargs)

    def save_checkpoints(self, ckpt_period=-1):
        """
        Save a checkpoint of the Optimizer
        
        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.save_ckpt_cbfn, kwargs)

    def monitor_state(self, ckpt_period=-1):
        """
        Monitor the state of the optimization.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        self.states = []
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.state_cbfn, kwargs)

    def monitor_shdom_iterations(self, ckpt_period=-1):
        """Monitor the number of SHDOM forward iterations.
        
        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'shdom iterations'
        }
        self.add_callback_fn(self.shdom_iterations_cbfn, kwargs)

    def monitor_scatterer_error(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor relative and overall mass error (epsilon, delta) as defined at:
          Amit Aides et al, "Multi sky-view 3D aerosol distribution recovery".
        
        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['{}/delta/{}', '{}/epsilon/{}']
        }
        self.add_callback_fn(self.scatterer_error_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})


    def monitor_scatter_plot(self, estimator_name, ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/scatter_plot/{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.scatter_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})


    def monitor_slices_plot(self, estimator_name, ground_truth, ckpt_period=-1, parameters='all'):
        """
        Monitor slices plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/3D_slices/{}',
            'parameters': parameters
        }
        self.add_callback_fn(self.slices_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_scatter_log_plot(self, estimator_name, ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/scatter_plot/{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.scatter_plot_log_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_horizontal_mean(self, estimator_name, ground_truth, ground_truth_mask=None, ckpt_period=-1):
        """
        Monitor horizontally averaged quantities and compare to ground truth over iterations.

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ground_truth_mask: shdom.GridData
            The ground-truth mask of the estimator
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/horizontal_mean/{}',
            'mask': ground_truth_mask
        }
        self.add_callback_fn(self.horizontal_mean_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_domain_mean(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor domain mean and compare to ground truth over iterations.

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/mean/{}'
        }
        self.add_callback_fn(self.domain_mean_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_images(self, measurements, ckpt_period=-1):
        """
        Monitor the synthetic images and compare to the acquired images
    
        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = measurements.images.copy() # if the measurments of type cloudct...., the images in grayscale.
        if(isinstance(measurements,shdom.CloudCT_setup.SpaceMultiView_Measurements)):
            sensor_type = measurements.sensor_type
            num_images = 0
            vmax_dict  = OrderedDict()
            
            for imager_index, images in acquired_images.items():
                num_images += len(images)
                vmax_dict[imager_index] = max([image.max() * 1.25 for image in images])
            
            vmin = min([v for index, v in vmax_dict.items()])
            
            for imager_index, images in acquired_images.items():
                tmp_max = max([image.max() * 1.25 for image in images])
                if(tmp_max > vmin ):
                    scale = vmin/tmp_max
                    acquired_images[imager_index] = [image * scale for image in images]
                    
            # after scaling, return the vmax to be the maximum over the images which was the minumum over the maximums.
            vmax = vmin
            acquired_images = list(chain(*acquired_images.values())) # Concatenating dictionary value lists. 
            
        else:
            sensor_type = measurements.camera.sensor.type
            num_images = len(acquired_images)

            if sensor_type == 'RadianceSensor':
                vmax = [image.max() * 1.25 for image in acquired_images]
            elif sensor_type == 'StokesSensor':
                vmax = [image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image in acquired_images]

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title':  ['Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_images_cbfn, kwargs)
        acq_titles = ['Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])

    def save_ckpt_cbfn(self, kwargs=None):
        """
        Callback function that saves checkpoints .

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        timestr = time.strftime("%H%M%S")
        path = os.path.join(self.tf_writer.logdir,  timestr + '.ckpt')
        self.optimizer.save_state(path)
        
    def loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.loss, self.optimizer.iteration)

    def state_cbfn(self, kwargs=None):
        """
        Callback function that is called for state monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        state = np.empty(shape=(0), dtype=np.float64)
        for estimator in self.optimizer.medium.estimators.values():
            for param in estimator.estimators.values():
                state = np.concatenate((state, param.get_state() / param.precondition_scale_factor))
        self.states.append(state)

    def estimated_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self.optimizer.iteration, self.optimizer.images, kwargs['title'], kwargs['vmax'])
    
    def shdom_iterations_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for shdom iteration monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.rte_solver.num_iterations, self.optimizer.iteration)
        
    def scatterer_error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)

            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)

            for parameter_name, parameter in est_scatterer.estimators.items():
                ground_truth = getattr(gt_scatterer, parameter_name)

                est_parameter_masked = copy.copy(parameter).resample(common_grid)
                est_parameter_masked.apply_mask(common_mask)
                est_param = est_parameter_masked.data.ravel()

                gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                gt_param_masked.apply_mask(common_mask)
                gt_param = gt_param_masked.data.ravel()

                delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param,1)
                self.tf_writer.add_scalar(kwargs['title'][0].format(scatterer_name, parameter_name), delta, self.optimizer.iteration)
                self.tf_writer.add_scalar(kwargs['title'][1].format(scatterer_name, parameter_name), epsilon, self.optimizer.iteration)

        
        
    def domain_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring domain averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            for parameter_name, parameter in est_scatterer.estimators.items():
                if parameter.type == 'Homogeneous':
                    est_param = parameter.data
                else:
                    est_param = parameter.data.mean()

                ground_truth = getattr(gt_scatterer, parameter_name)
                if ground_truth.type == 'Homogeneous':
                    gt_param = ground_truth.data
                else:
                    gt_param = ground_truth.data.mean()

                self.tf_writer.add_scalars(
                    main_tag=kwargs['title'].format(scatterer_name, parameter_name),
                    tag_scalar_dict={'estimated': est_param, 'true': gt_param},
                    global_step=self.optimizer.iteration
                )

    def horizontal_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring horizontal averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)

            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)

            for parameter_name, parameter in est_scatterer.estimators.items():
                ground_truth = getattr(gt_scatterer, parameter_name)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)


                    est_parameter_masked = copy.deepcopy(parameter).resample(common_grid)
                    est_parameter_masked.apply_mask(common_mask)
                    est_param = est_parameter_masked.data
                    est_param[np.bitwise_not(common_mask.data)] = np.nan
                    est_param = np.nan_to_num(np.nanmean(est_param,axis=(0,1)))

                    gt_param_masked = copy.deepcopy(ground_truth).resample(common_grid)
                    gt_param_masked.apply_mask(common_mask)
                    gt_param = gt_param_masked.data
                    gt_param[np.bitwise_not(common_mask.data)] = np.nan
                    gt_param = np.nan_to_num(np.nanmean(gt_param,axis=(0,1)))

                fig, ax = plt.subplots()
                ax.set_title('{} {}'.format(scatterer_name, parameter_name), fontsize=16)
                ax.plot(est_param, est_scatterer.grid.z, label='Estimated')
                ax.plot(gt_param, est_scatterer.grid.z, label='True')
                ax.legend()
                ax.set_ylabel('Altitude [km]', fontsize=14)
                self.tf_writer.add_figure(
                    tag=kwargs['title'].format(scatterer_name, parameter_name),
                    figure=fig,
                    global_step=self.optimizer.iteration
                )
        
    def scatter_plot_log_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
    
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)
    
            parameters = est_scatterer.estimators.keys() if kwargs['parameters']=='all' else kwargs['parameters']
            for parameter_name in parameters:
                if parameter_name not in est_scatterer.estimators.keys():
                    continue
                if parameter_name == 'lwc':
                    parameter = est_scatterer.estimators[parameter_name]
                    ground_truth = getattr(gt_scatterer, parameter_name)
        
                    est_parameter_masked = copy.copy(parameter).resample(common_grid)
                    est_parameter_masked.apply_mask(common_mask)
                    est_param = est_parameter_masked.data.ravel()
        
                    gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                    gt_param_masked.apply_mask(common_mask)
                    gt_param = gt_param_masked.data.ravel()
        
                    rho = np.corrcoef(est_param, gt_param)[1, 0]
                    num_params = gt_param.size
                    rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                    max_val = max(gt_param.max(), est_param.max())
                    fig, ax = plt.subplots()
                    ax.set_title(r'{} {}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(scatterer_name, parameter_name, 100 * kwargs['percent'], rho),
                                     fontsize=16)
                    ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='r')
                    ax.plot(ax.get_ylim(), ax.get_ylim(), c='k', ls='--')
                    
                    ax.set_yscale('log')
                    ax.set_xscale('log')
                    max_val = np.log10(max_val)
                    
                    ax.set_xlim([0, 1.1*max_val])
                    ax.set_ylim([0, 1.1*max_val])
                    ax.set_ylabel('Estimated', fontsize=14)
                    ax.set_xlabel('True', fontsize=14)
                    
                    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
                    
                    ax.set_aspect('equal')                    
        
                    self.tf_writer.add_figure(
                            tag=kwargs['title'].format(scatterer_name, parameter_name),
                            figure=fig,
                            global_step=self.optimizer.iteration
                        )
                    
    def slices_plot_cbfn(self, kwargs): 
        """
        Callback function for monitoring in 3D.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        def show_volume_slices(volume,X, Y, Z, SLICE_X = 0, SLICE_Y = 0, SLICE_Z = 0):
            # visualize volume:
            MAXI = volume.max()
            h = mlab.pipeline.scalar_field(X, Y, Z, volume)
            v = mlab.pipeline.volume(h,vmin=0.0,vmax=MAXI)
        
            ipw_x = mlab.pipeline.image_plane_widget(h, plane_orientation='x_axes',vmin=0.0,vmax=MAXI)
            ipw_x.ipw.slice_position = SLICE_X
            ipw_x.ipw.reslice_interpolate = 'nearest_neighbour'
            ipw_x.ipw.texture_interpolate = False
            ipw_y = mlab.pipeline.image_plane_widget(h, plane_orientation='y_axes',vmin=0.0,vmax=MAXI)
            ipw_y.ipw.slice_position = SLICE_Y
            ipw_y.ipw.reslice_interpolate = 'nearest_neighbour'
            ipw_y.ipw.texture_interpolate = False
            ipw_z = mlab.pipeline.image_plane_widget(h, plane_orientation='z_axes',vmin=0.0,vmax=MAXI)
            ipw_z.ipw.slice_position = SLICE_Z
            ipw_z.ipw.reslice_interpolate = 'nearest_neighbour'
            ipw_z.ipw.texture_interpolate = False
        
            color_bar = mlab.colorbar(orientation='vertical', nb_labels=5)
            
            mlab.view(40, 50)
            
            mlab.outline(color = (1, 1, 1))  # box around data axes
            
        for scatterer_name, gt_scatterer in self._ground_truth.items():
    
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)
    
            parameters = est_scatterer.estimators.keys() if kwargs['parameters']=='all' else kwargs['parameters']
            for parameter_name in parameters:
                if parameter_name not in est_scatterer.estimators.keys():
                    continue
                parameter = est_scatterer.estimators[parameter_name]
                ground_truth = getattr(gt_scatterer, parameter_name)
    
                est_parameter_masked = copy.copy(parameter).resample(common_grid)
                est_parameter_masked.apply_mask(common_mask)
                est_param_3d = est_parameter_masked.data
    
                gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                gt_param_masked.apply_mask(common_mask)
                gt_param_3d = gt_param_masked.data
    
                fig, ax = plt.subplots(ncols=2)
                
                x = common_grid.x
                y = common_grid.y
                z = common_grid.z
                X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
                
                SLICE_X = x[gt_param_3d.sum(axis = (1,2)).argmax()]
                SLICE_Y = y[gt_param_3d.sum(axis = (0,2)).argmax()]
                SLICE_Z = z[gt_param_3d.sum(axis = (0,1)).argmax()]
                # gt:
                show_volume_slices(gt_param_3d, X, Y, Z, SLICE_X, SLICE_Y, SLICE_Z)
                mlab.title('True')
                frame_gt = mlab.screenshot(antialiased=True) 
                # estimated:
                show_volume_slices(est_param_3d, X, Y, Z, SLICE_X, SLICE_Y, SLICE_Z)
                mlab.title('Estimated')
                frame_est = mlab.screenshot(antialiased=True) 
                
                ax[0].set_title('True',fontsize=16)
                ax[0].imshow(frame_gt)
                
                ax[1].set_title('Estimated',fontsize=16)
                ax[1].imshow(frame_est)     
    
                self.tf_writer.add_figure(
                        tag=kwargs['title'].format(scatterer_name, parameter_name),
                        figure=fig,
                        global_step=self.optimizer.iteration
                    )
                
        
    def scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
    
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)
    
            parameters = est_scatterer.estimators.keys() if kwargs['parameters']=='all' else kwargs['parameters']
            for parameter_name in parameters:
                if parameter_name not in est_scatterer.estimators.keys():
                    continue
                parameter = est_scatterer.estimators[parameter_name]
                ground_truth = getattr(gt_scatterer, parameter_name)
    
                est_parameter_masked = copy.copy(parameter).resample(common_grid)
                est_parameter_masked.apply_mask(common_mask)
                est_param = est_parameter_masked.data.ravel()
    
                gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                gt_param_masked.apply_mask(common_mask)
                gt_param = gt_param_masked.data.ravel()
    
                rho = np.corrcoef(est_param, gt_param)[1, 0]
                num_params = gt_param.size
                rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                max_val = max(gt_param.max(), est_param.max())
                fig, ax = plt.subplots()
                
                # ---- to add the colors to spots:
                x = common_grid.x
                y = common_grid.y
                z = common_grid.z
                X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
                Z = Z.ravel()
                ax.set_title(r'{} {}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(scatterer_name, parameter_name, 100 * kwargs['percent'], rho),
                                 fontsize=16)
                #ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
                im = ax.scatter(gt_param[rand_ind], est_param[rand_ind], s = 15, c = Z[rand_ind])
                ax.set_xlim([0, 1.1*max_val])
                ax.set_ylim([0, 1.1*max_val])
                ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                ax.set_ylabel('Estimated', fontsize=14)
                ax.set_xlabel('True', fontsize=14)
                # colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)                
    
                self.tf_writer.add_figure(
                        tag=kwargs['title'].format(scatterer_name, parameter_name),
                        figure=fig,
                        global_step=self.optimizer.iteration
                    )
        
        

    def write_image_list(self, global_step, images, titles, vmax=None):
        """
        Write an image list to tensorboardX.
    
        Parameters
        ----------
        global_step: integer,
            The global step of the optimizer.
        images: list
            List of images to be logged onto tensorboard.
        titles: list
            List of strings that will title the corresponding images on tensorboard.
        vmax: list or scalar, optional
            List or a single of scaling factor for the image contrast equalization
        """
        if np.isscalar(vmax) or vmax is None:
            vmax = [vmax]*len(images)        
    
        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))
    
        for image, title, vm in zip(images, titles, vmax):

            # for polarization
            if image.ndim == 4:
                stoke_title = ['V', 'U', 'Q', 'I']
                for v, stokes in zip(vm, image):
                    self.tf_writer.add_images(
                        tag=title + '/' + stoke_title.pop(),
                        img_tensor=(np.repeat(np.expand_dims(stokes, 2), 3, axis=2) / v),
                        dataformats='HWCN',
                        global_step=global_step
                    )

            # for polychromatic
            elif image.ndim == 3:
                self.tf_writer.add_images(
                    tag=title, 
                    img_tensor=(np.repeat(np.expand_dims(image, 2), 3, axis=2) / vm),
                    dataformats='HWCN',
                    global_step=global_step
                )
            # for monochromatic
            else:
                self.tf_writer.add_image(
                    tag=title, 
                    img_tensor=(image / vm),
                    dataformats='HW',
                    global_step=global_step
                )

    @property
    def callback_fns(self):
        return self._callback_fns

    @property
    def dir(self):
        return self._dir

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def tf_writer(self):
        return self._tf_writer
    

class SpaceCarver(object):
    """
    SpaceCarver object recovers the convex hull of the cloud based on multi-view sensor geometry and pixel segmentation.
    
    Parameters
    ----------
    measurements: shdom.Measurements
        A measurements object storing the images and sensor geometry
    """
    def __init__(self, measurements):
    
        self._rte_solver = shdom.RteSolver(shdom.SceneParameters(), shdom.NumericalParameters())
        
        self._measurements = measurements
        
        if(isinstance(measurements,shdom.CloudCT_setup.SpaceMultiView_Measurements)):
            self._projections = []
            self._images = []
            # relating to CloudCT multi-view setup with different imagers.
            CloudCT_geometry_and_imagers = measurements.setup  
            images_dict = measurements.images.copy()
            
            imagers_channels = measurements.get_channels_of_imagers() # A list, it is the central wavelength of al imagers. 
            for imager_index, wavelength in enumerate(imagers_channels):
                acquired_images = images_dict[imager_index]
                CloudCT_projection = CloudCT_geometry_and_imagers[imager_index]  
                
                self._projections +=  CloudCT_projection.projection_list
                self._images += acquired_images.copy()
         
        else:
            
            if isinstance(measurements.camera.projection, shdom.MultiViewProjection):
                self._projections = measurements.camera.projection.projection_list
            else:
                self._projections = [measurements.camera.projection]
            self._images = measurements.images

    def carve(self, grid, thresholds, agreement=0.75, PYTHON_SPACE_CURVE = False):
        """
        Carves out the cloud geometry on the grid. 
        A threshold on radiances is used to produce a pixel mask and preform space carving.
        
        Parameters
        ----------
        grid: shdom.Grid
            A grid object.
        thresholds: list or float
            Either a constant threshold or a list of len(thresholds)=num_projections is used as for masking.
        agreement: float
            the precentage of pixels that should agree on a cloudy voxels to set it to True in the mask
        
        PYTHON_SPACE_CURVE - bool, If True, use Python implmentation by vadim for space curving. It just gives thiner mask than the original
        space curving (fortran implmentation) and maybe less numeric problems of distant views like in CloudCT.
        
        Returns
        -------
        mask: shdom.GridData object
            A boolean mask with True marking cloudy voxels and False marking non-cloud region.

        Notes
        -----
        Currently ignores stokes/multispectral measurements and uses only I component and the last channel to retrieve a cloud mask.
        """
        
        VIS_VOLUME = False # for debug purposes
        
        self._rte_solver.set_grid(grid)
        volume = np.zeros((grid.nx, grid.ny, grid.nz))
        
        thresholds = np.array(thresholds)
        if thresholds.size == 1:
            thresholds = np.repeat(thresholds, len(self._images))
        else:
            assert thresholds.size == len(self._images), 'thresholds (len={}) should be of the same' \
                   'length as the number of images (len={})'.format(thresholds.size,  len(self._images))
          
          
        #-----------------
        f, axarr = plt.subplots(1, len(self._images), figsize=(20, 20))
        for ax, image in zip(axarr, self._images):
            img = image.copy()
            img[image < thresholds[0]] = 0
            ax.imshow(img)
            ax.invert_xaxis() 
            ax.invert_yaxis() 
            ax.axis('off')
        
        plt.show()
        #-----------------------
            
        for view_index, (projection, image, threshold) in enumerate(zip(self._projections, self._images, thresholds)):

            if(self._measurements.sensor_type == 'StokesSensor'):
                image = image[0]
            """
            TODO - if imager has multiple cahnnels this must be adjusted.
            """    
            if(projection.samples_per_pixel > 1):
                samples_per_pixel = projection.samples_per_pixel
                num_channels = 1
                image_mask = image > threshold
                mask_pixels = np.repeat(image_mask.reshape((-1, num_channels), order='F'), samples_per_pixel)
                projection = projection[mask_pixels == 1]
                
                    
            else:
                
                if(not isinstance(self._measurements,shdom.CloudCT_setup.SpaceMultiView_Measurements)):
                    # original usage;
                    if self._measurements.num_channels > 1:
                        image = image[..., -1]
                    if self._measurements.camera.sensor.type == 'StokesSensor':
                        image = image[0]
    
                image_mask = image > threshold
                    
                projection = projection[image_mask.ravel(order='F') == 1]
            
            if(PYTHON_SPACE_CURVE):
                # use space curving for CloudCT (I think it is more stable):
                print('Processing: building mask by space curving of view {} from {} total views'.format(view_index,len(self._images)))
                carved_volume = self.CloudCT_space_carve(projection,grid)
                volume += carved_volume
                
                file_name = 'volume_{}'.format(view_index)+'.mat'
                sio.savemat(file_name, {'vol':carved_volume})                
                    
            else:
                
                carved_volume = core.space_carve(
                    nx=grid.nx,
                    ny=grid.ny,
                    nz=grid.nz,
                    npts=self._rte_solver._npts,
                    ncells=self._rte_solver._ncells,
                    gridptr=self._rte_solver._gridptr,
                    neighptr=self._rte_solver._neighptr,
                    treeptr=self._rte_solver._treeptr,
                    cellflags=self._rte_solver._cellflags,
                    bcflag=self._rte_solver._bcflag,
                    ipflag=self._rte_solver._ipflag,
                    xgrid=self._rte_solver._xgrid,
                    ygrid=self._rte_solver._ygrid,
                    zgrid=self._rte_solver._zgrid,
                    gridpos=self._rte_solver._gridpos,
                    camx=projection.x,
                    camy=projection.y,
                    camz=projection.z,
                    cammu=projection.mu,
                    camphi=projection.phi,
                    npix=projection.npix,
                )
                volume += carved_volume.reshape(grid.nx, grid.ny, grid.nz)
        
        volume = volume * 1.0 / len(self._images)
        
        if(VIS_VOLUME):
            # visualize volume:
            try:
                import mayavi.mlab as mlab
        
            except:
                raise Exception("Make sure you installed mayavi")

            
            MAXI = 1
            show_vol = volume > agreement
            show_vol = np.multiply(show_vol, 1, dtype= np.int16) 
            
            # I was testing holes with the following two lines:
            #show_vol = np.zeros_like(volume)
            #show_vol[inds[:,0],inds[:,1],inds[:,2]] = volume[inds[:,0],inds[:,1],inds[:,2]]    
                
            h = mlab.pipeline.scalar_field(show_vol)
            v = mlab.pipeline.volume(h,vmin=0.0,vmax=MAXI)
        
            ipw_x = mlab.pipeline.image_plane_widget(h, plane_orientation='x_axes',vmin=0.0,vmax=MAXI)
            ipw_x.ipw.reslice_interpolate = 'linear'
            ipw_x.ipw.texture_interpolate = False
            ipw_y = mlab.pipeline.image_plane_widget(h, plane_orientation='y_axes',vmin=0.0,vmax=MAXI)
            ipw_y.ipw.reslice_interpolate = 'linear'
            ipw_y.ipw.texture_interpolate = False
            ipw_z = mlab.pipeline.image_plane_widget(h, plane_orientation='z_axes',vmin=0.0,vmax=MAXI)
            ipw_z.ipw.reslice_interpolate = 'linear'
            ipw_z.ipw.texture_interpolate = False
        
            color_bar = mlab.colorbar(orientation='vertical', nb_labels=5)
            mlab.outline(color = (1, 1, 1))  # box around data axes
            
            mlab.show()
         
        show_vol = volume > agreement
        show_vol = np.multiply(show_vol, 1, dtype= np.int16)         
        file_name = 'The_mask.mat'
        sio.savemat(file_name, {'vol':show_vol}) 
        
        mask = GridData(grid, volume > agreement) 
        return mask

    @property
    def grid(self):
        return self._grid
    
    @property
    def images(self):
    	return self._images
    
    def CloudCT_space_carve(self,projection,grid):
        """
        Python implmentation by vadim for space curving. It just gives thiner mask than the original
        space curving (fortran implmentation) and maybe less numeric problems of distant views like in CloudCT.
        
        """
        carved_volume = np.zeros((grid.nx, grid.ny, grid.nz))
        for pix in range(projection.nrays):
            z_c = -projection.mu[pix]
            x_c = -np.sin(np.arccos(-z_c))*np.cos(projection.phi[pix]) 
            y_c = -np.sin(np.arccos(-z_c))*np.sin(projection.phi[pix])  
    
            start_point = np.array([projection.x[pix], projection.y[pix], projection.z[pix]])
            direction = np.array([x_c,y_c,z_c])
            voxels, lengths = LengthRayInGrid(start_point,direction,grid)
            if np.isscalar(voxels):
    
                if (voxels==-1)  or (lengths==-1):
    
                    continue
    
            indices = np.unravel_index(voxels, carved_volume.shape)
            carved_volume[indices] = 1                            
    
        return carved_volume         


class LocalOptimizer(object):
    """
    The LocalOptimizer class takes care of the under the hood of the optimization process.
    To run the optimization the following methods should be called:
       [required] optimizer.set_measurements()
       [required] optimizer.set_rte_solver()
       [required] optimizer.set_medium_estimator() 
       [optional] optimizer.set_writer()

    Parameters
    ----------
    options: dict
        The option dictionary for the optimizer
    n_jobs: int, default=1
        The number of jobs to divide the gradient computation into.
    init_solution: bool
        True will re-initialize the solution process every iteration.
        False will use the previous step RTE solution to initialize the current RTE solution.
    method: str, default='L-BFGS-B'
        The optimizer solution method: 'L-BFGS-B', 'TNC'

    Notes
    -----
    Currently only L-BFGS-B optimization method is supported.
    For documentation:
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    """
    def __init__(self, method, options={}, n_jobs=1, init_solution=True, smoothness_ratio = 0.1):
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        if method not in ['L-BFGS-B', 'TNC']:
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self._options = options
        self._smoothness_ratio = smoothness_ratio
        
    def set_measurements(self, measurements):
        """
        Set the measurements (data-fit constraints)
        
        Parameters
        ----------
        measurements: shdom.Measurements or shdom.CloudCT_setup.SpaceMultiView_Measurements
            A measurements object storing the acquired images and sensor geometry
        """
        self._measurements = measurements
    
    def set_medium_estimator(self, medium_estimator):
        """
        Set the MediumEstimator for the optimizer.
        
        Parameters
        ----------
        medium_estimator: shdom.MediumEstimator
            The MediumEstimator
        """
        self._medium = medium_estimator

    def set_rte_solver(self, rte_solver):
        """
        Set the RteSolver for the SHDOM iterations.
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver
        """
        if isinstance(rte_solver, shdom.RteSolverArray):
            self._rte_solver = rte_solver
        else:
            self._rte_solver = shdom.RteSolverArray([rte_solver])

    def set_writer(self, writer):
        """
        Set a log writer to upload summaries into tensorboard.
        
        Parameters
        ----------
        writer: shdom.SummaryWriter
            Wrapper for the tensorboardX summary writer.
        """
        self._writer = writer
        if writer is not None:
            self._writer.attach_optimizer(self)   
	    
	    
	    
    def Layer_smoother(self,state,smoothness_ratio):
	"""
	Vadim added on 9/12/2020
	Calculate regularization (Laplacian in x, y axes) cost and gradient.

	Parameters
	----------
	state: np.array(shape=(self.num_parameters, dtype=np.float64)
	    The current state vector

	Returns
	-------
	loss: np.float64
	    The total loss accumulated over all pixels
	gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
	    The gradient of the objective function with respect to the state parameters

	"""

	# ----------------------------------------------------------------
	# ------- extract reff state in its geometry (1D or 3D):----------
	# ----------------------------------------------------------------
	"""
	Use regularization type = Laplacian.

	"""
	regularization_grad = np.zeros_like(state)
	regularization_cost = 0

	cloud_estimator = self.medium.estimators['cloud']
	if 'reff' not in cloud_estimator.estimators.keys():
	    return regularization_cost, regularization_grad

	# ----------------------------------------------------------------
	# ------- extract reff state in its geometry (1D or 3D):----------
	# ----------------------------------------------------------------        
	states_indexes = np.split(np.arange(state.size), np.cumsum(cloud_estimator.num_parameters[:-1]))
	reff_estimator = cloud_estimator.estimators['reff']
	lwc_estimator = cloud_estimator.estimators['lwc']
	reff_data = reff_estimator.data.copy()
	lwc_data = lwc_estimator.data.copy()

	if reff_estimator.grid.type == '1D':
	    print("No Layer smoothing: The grid is 1D!") 
	    reff_data_1d = reff_data[reff_estimator.mask.data]
	    return regularization_cost, regularization_grad
	# ----------------------------------------------------------------
	# ------- functions:----------------------------------------------
	# ---------------------------------------------------------------- 
	# smoothness_ratio and reff_data are known to that function
	def f2d(X,mask,scale):
	    X = np.reshape(X,reff_data.shape[:2])
	    scale = np.reshape(scale,reff_data.shape[:2])

	    dilated_layer = ndimage.grey_dilation(X, size=(6,6))
	    dilated_layer[mask] = X[mask]

	    norm = np.linalg.norm(scale*laplacian(dilated_layer)*mask)**2

	    return smoothness_ratio*norm        

	def g2d(X,mask,scale):
	    X = np.reshape(X,reff_data.shape[:2])
	    scale = np.reshape(scale,reff_data.shape[:2])

	    dilated_layer = ndimage.grey_dilation(X, size=(6,6))
	    dilated_layer[mask] = X[mask]

	    grad = 2 * smoothness_ratio * laplacian((scale**2)*laplacian(dilated_layer)*mask)*mask

	    return grad.flatten()

	def f(X,mask_3d,scale_3d=None):
	    # X is flatten vector, it should be 3d matrix
	    # mask is 3D bool matrix
	    X = np.reshape(X,mask_3d.shape)
	    if scale_3d is None:
		scale_3d = np.ones_like(mask_3d)

	    loss = 0
	    for altitude_index in range(mask_3d.shape[2]):
		mask = mask_3d[:,:,altitude_index].copy()
		scale = scale_3d[:,:,altitude_index].copy()
		loss_layer = f2d(X[:,:,altitude_index].flatten(),mask,scale.flatten())
		loss += loss_layer

	    print('regularization loss is {}'.format(loss))

	    return loss

	def g(X,mask_3d,scale_3d=None):
	    # X is flatten vector, it should be 3d matrix
	    # mask is 3D bool matrix
	    X = np.reshape(X,mask_3d.shape)
	    if scale_3d is None:
		scale_3d = np.ones_like(mask_3d)

	    gradient = np.zeros_like(reff_data)

	    for altitude_index in range(mask_3d.shape[2]):
		mask = mask_3d[:,:,altitude_index].copy()
		scale = scale_3d[:,:,altitude_index].copy()
		gradient[:,:,altitude_index] = g2d(X[:,:,altitude_index].flatten(),mask,scale.flatten()).reshape(mask_3d.shape[:2])

	    return gradient.flatten()  


	# ---------------------------------------------------------------------------------
	# ---------------------------------------------------------------------------------
	# ---------------------------------------------------------------------------------
	mask = reff_estimator.mask.data
	# At this point the mask and data is 3D.
	assert reff_estimator.grid.type == '3D', "Unsupported grid type {}".format(reff_estimator.grid.type)        
	# Preconditioning:
	reff_data*= reff_estimator.precondition_scale_factor
	#lwc_data*= lwc_estimator.precondition_scale_factor # I don't know if that is necessary
	lwc_scale = lwc_data/lwc_data.max()
	data_smooth = reff_data
	# use lwc_scale (operator M) to scale the gradient:
	# cost (MLx)'(MLx) = |MLx|^2, where x is a layer of the state.
	# gradient 2*(L'M'MLx)
	regularization_cost = f(data_smooth.flatten(),mask,lwc_scale.flatten())
	grad_3d = g(data_smooth.flatten(),mask,lwc_scale.flatten()).reshape(reff_data.shape)
	regularization_grad[states_indexes[1]] = grad_3d[mask]

	return regularization_cost, regularization_grad

    
    def profile_smoother(self,state,smoothness_ratio):
	"""
	Vadin added on 12/12/2020 - Not in use yet.
	
	"""
	regularization_grad = np.zeros_like(state)
	regularization_cost = 0
    
	if self.smoothness_ratio > 0:
        
	    cloud_estimator = self.medium.estimators['cloud']
	    if 'reff' in cloud_estimator.estimators.keys():
		states_indexes = np.split(np.arange(state.size), np.cumsum(cloud_estimator.num_parameters[:-1]))
		reff_estimator = cloud_estimator.estimators['reff']
		reff_data = reff_estimator.data.copy()
		regularization_grad = np.zeros_like(state)
    
		if reff_estimator.grid.type == '1D':
		    reff_data_1d = reff_data[reff_estimator.mask.data].copy()
		elif reff_estimator.grid.type == '3D':
		    # masked mean
		    masked_mean = np.ma.masked_equal(reff_data, 0).mean(axis=0).mean(axis=0)
		    reff_data_1d = masked_mean.data                  
		else:
		    raise Exception('Unsupported grid type.')
    
		reff_data_1d*= reff_estimator.precondition_scale_factor
		regularization_grad[states_indexes[1]] = self.smoothness_ratio *2* laplacian(laplacian(reff_data_1d))
		regularization_cost = self.smoothness_ratio * np.linalg.norm(laplacian(reff_data_1d))**2
	
	return regularization_cost, regularization_grad
	# ----------------------------------------------------------------
	
	
    def objective_fun(self, state):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        #if(self.iteration % 5 == 0):
            #state = 5*state
            
        self.set_state(state,self.iteration)
        # ----------------------------------------------------------------
        # ------- extract reff state in its geometry (1D or 3D):----------
        # ----------------------------------------------------------------
        """
        Use regularization type = Laplacian.
        
        """
	regularization_cost, regularization_grad = self.Layer_smoother(state,self.smoothness_ratio)  
        
        gradient, loss, images, debug_gradients = self.medium.compute_gradient(
            rte_solvers=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs, iteration = self.iteration
        )
        print('cost:')
        print(loss, regularization_cost)
        print(20*'-')
        print('state, gradient:')
        print(state, gradient)
        print(20*'-')
        
        self._loss = loss
        self._images = images
        
        if(self.iteration == 15):
            cost_tracker_path = os.path.join(self.writer.dir,'cost_itr_15.txt')
            with open(cost_tracker_path, 'a') as file_object:
                file_object.write('{}\n'.format(loss))            
        
        # vadim save gradients for each iteration for debug:
        SAVE_GRADIENTS_DB = False
        if(SAVE_GRADIENTS_DB):
            gredient_save_dir = self.writer.dir + '_gradients'  
            
            if not os.path.exists(gredient_save_dir):
                os.mkdir(gredient_save_dir)
                
            scatterers = self.medium.estimators.keys()
            
            for scatterer_name in scatterers:
                scatterer = self.medium.get_scatterer(name=scatterer_name)
                file_name = 'iteration_{}.txt'.format(self.iteration) 
                file_name  = os.path.join(self.writer.dir,file_name)
                
                #lwc = scatterer.lwc.resample(scatterer.grid)
                #reff = scatterer.reff.resample(scatterer.grid)
                #veff = scatterer.veff.resample(scatterer.grid)               
                #droplets = shdom.MicrophysicalScatterer(lwc, reff, veff)
                
                #scatterer_copy = copy.deepcopy(scatterer)
                #scatterer_copy.lwc = scatterer.lwc.resample(scatterer.grid)
                #scatterer_copy.reff = scatterer.reff.resample(scatterer.grid)
                #scatterer_copy.veff = scatterer.veff.resample(scatterer.grid)                
                #scatterer_copy.save_to_csv(file_name, comment_line='from some minimization process')
                
                for index,parameter_name in enumerate(scatterer.estimators.keys()):
                    filename = 'gradient_of_{}_at_iteration_{}.mat'.format(parameter_name,self.iteration) 
                    grad_save = debug_gradients[index]
                    estimator = scatterer.estimators[parameter_name]
                    sio.savemat(os.path.join(gredient_save_dir,filename), {'data':grad_save})
                    
                    filename = '{}_at_iteration_{}.mat'.format(parameter_name,self.iteration)                                        
                    sio.savemat(os.path.join(gredient_save_dir,filename), {'data':estimator})
        
        loss += regularization_cost
        gradient += regularization_grad
        return loss, gradient
            
    def callback(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any). 
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector
        """
        self._iteration += 1
        
        # Writer callback functions
        if self.writer is not None:
            for callbackfn, kwargs in zip(self.writer.callback_fns, self.writer.kwargs):
                time_passed = time.time() - kwargs['ckpt_time']
                if time_passed > kwargs['ckpt_period']:
                    kwargs['ckpt_time'] = time.time()
                    callbackfn(kwargs)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        if self.iteration == 0:
            self.init_optimizer()

        result = minimize(fun=self.objective_fun,
                          x0=self.get_state(),
                          method=self.method,
                          jac=True,
                          bounds=self.get_bounds(),
                          options=self.options,
                          callback=self.callback)
        return result
    
    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """

        assert self.rte_solver.num_solvers == self.measurements.num_channels == self.medium.num_wavelengths, \
            'RteSolver has {} solvers, Measurements have {} channels and Medium has {} wavelengths'.format(self.rte_solver.num_solvers, self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters
        
    def get_bounds(self):
        """
        Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        return self.medium.get_bounds()
    
    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return self.medium.get_state()
    
    def set_state(self, state, optimization_iteration = None):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        
        self.medium.set_state(state, optimization_iteration)
        self.rte_solver.set_medium(self.medium)
        if self._init_solution is False:
            self.rte_solver.make_direct()
        self.rte_solver.solve(maxiter=100, init_solution=self._init_solution, verbose=False)
        
    def save_state(self, path):
        """
        Save Optimizer state to file.
        
        Parameters
        ----------
        path: str,
            Full path to file. 
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_state(), -1))
        file.close()

    def load_state(self, path):
        """
        Load Optimizer from file.
        
        Parameters
        ----------
        path: str,
            Full path to file. 
        """
        file = open(path,'rb')
        data = file.read()
        file.close()        
        state = pickle.loads(data)
        self.set_state(state)

    @property
    def method(self):
        return self._method

    @property
    def smoothness_ratio(self):
        return self._smoothness_ratio

    @property
    def options(self):
        return self._options

    @property
    def rte_solver(self):
        return self._rte_solver
    
    @property
    def medium(self):
        return self._medium    
    
    @property
    def measurements(self):
        return self._measurements
    
    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def writer(self):
        return self._writer      
    
    @property
    def iteration(self):
        return self._iteration

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def loss(self):
        return self._loss        
    
    @property
    def images(self):
        return self._images


class ProximalProjection(object):
    """TODO"""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self):
        if self.optimizer.writer is not None:
            self.optimizer.writer.attach_optimizer(self.optimizer)
        self.optimizer.minimize()


class LocalOptimizerADMM(LocalOptimizer):
    """"TODO"""
    def __init__(self, method, options={}, n_jobs=1):
        self._options = options
        self._n_jobs = n_jobs
        self._method = method
        self._proximal_projections = []

    def init_optimizer(self):
        """TODO"""
        self._iter = 0
        if self.medium.num_estimators > 1:
            raise NotImplementedError('Multiple medium estimators not implemented')

        scatterer_estimator = next(iter(self.medium.estimators.values()))
        for param, param_estimator in scatterer_estimator.estimators.items():
            optimizer = shdom.LocalOptimizer(method=self.method, options=self.options, n_jobs=self.n_jobs)
            optimizer.set_measurements(self.measurements)
            optimizer.set_rte_solver(self.rte_solver)
            optimizer.set_writer(self.writer)
            medium_estimator = shdom.MediumEstimator(
                grid=self.medium.grid,
                loss_type=self.medium._loss_type,
                exact_single_scatter=self.medium._exact_single_scatter,
                stokes_weights=self.medium._stokes_weights
            )

            for name, scatterer in self.medium.scatterers.items():
                if scatterer == scatterer_estimator:
                    if param == 'lwc':
                        lwc = scatterer.lwc
                        reff = shdom.GridData(scatterer.reff.grid, scatterer.reff.data)
                        veff = shdom.GridData(scatterer.veff.grid, scatterer.veff.data)
                    elif param == 'reff':
                        lwc = shdom.GridData(scatterer.lwc.grid, scatterer.lwc.data)
                        reff = scatterer.reff
                        veff = shdom.GridData(scatterer.veff.grid, scatterer.veff.data)
                    elif param == 'veff':
                        lwc = shdom.GridData(scatterer.lwc.grid, scatterer.lwc.data)
                        reff = shdom.GridData(scatterer.reff.grid, scatterer.reff.data)
                        veff = scatterer.veff
                    proximal_scatterer = shdom.MicrophysicalScattererEstimator(scatterer.mie, lwc, reff, veff)
                    medium_estimator.add_scatterer(proximal_scatterer, name)
                else:
                    medium_estimator.add_scatterer(scatterer, name)

            optimizer.set_medium_estimator(medium_estimator)
            self.proximal_projections.append(shdom.ProximalProjection(optimizer))

    def minimize(self):
        """TODO"""
        self.init_optimizer()
        iter = 0
        while (iter < maxiter):
            [proximal() for proximal in self.proximal_projections]
            iter += 1

    @property
    def proximal_projections(self):
        return self._proximal_projections

class GlobalOptimizer(object):
    """
    The GlobalOptimizer class takes care of the under the hood of the global optimization process.
    To run the optimization a local optimizer should be set (see set_local_optimizer method)

    Parameters
    ----------
    local_optimizer: shdom.LocalOptimizer, optional
        A local optimizer object, after all initializations (see LocalOptimizer class)
    method: str
        The global optimization method.

    Notes
    -----
    Currently only basin-hopping global optimization is supported.

    For documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
    """
    def __init__(self, local_optimizer=None, method='basin-hopping'):
        self._iteration = 1
        self._best_minimum_f = np.Inf
        self._best_minimum_iteration = 0
        self._best_minimum_x = None
        self._loss = None
        self._take_step = None
        self._tf_writer = None
        if method != 'basin-hopping':
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self.set_local_optimizer(local_optimizer)

    def set_local_optimizer(self, local_optimizer):
        """
        Set the local optimizer.

        Parameters
        ----------
        local_optimizer: shdom.LocalOptimizer
            A local optimizer object, after all initializations (see LocalOptimizer class)
        """
        self._local_optimizer = local_optimizer
        self._take_step = RandomStep(local_optimizer.medium)
        self.init_local_optimizer()

    def minimize(self, niter_success, T=1e-3, maxiter=100, stepsize=0.5, interval=10, disp=True):
        """
        Global minimization with respect to the parameters defined.

        Parameters
        ----------
        niter_success: int
            Stop the run if the global minimum candidate remains the same for this number of iterations.
        T: float
            The “temperature” parameter for the accept or reject criterion.
            Higher “temperatures” mean that larger jumps in function value will be accepted.
            For best results T should be comparable to the separation (in function value) between local minima
        maxiter : int,
            The number of basin hopping iterations
        stepsize: float,
            Maximum step size for use in the random displacement. See RandomStep object for more info.
        interval: int,
            interval for how often to update the stepsize
        disp: bool
            Display information of the optimization process.
        """
        if self.method == 'basin-hopping':
            result = basinhopping(func=self.local_optimizer.objective_fun,
                                  x0=self.local_optimizer.get_state(),
                                  minimizer_kwargs=self.local_minimizer_kwargs,
                                  disp=disp,
                                  niter=maxiter,
                                  take_step=self.take_step,
                                  stepsize=stepsize,
                                  callback=self.callback,
                                  T=T,
                                  interval=interval,
                                  niter_success=niter_success)
        return result

    def init_local_optimizer(self):
        """Initialize the local optimizer and writer (if any)."""
        self.local_optimizer.init_optimizer()
        self._best_minimum_x = self.local_optimizer.get_state()
        local_options = self.local_optimizer.options
        local_options['disp'] = False
        self._local_minimizer_kwargs = {
            'method': self.local_optimizer.method,
            'jac': True,
            'bounds': self.local_optimizer.get_bounds(),
            'options': local_options,
            'callback': self.local_optimizer.callback
        }

        # If local writer exists, modify checkpoint saving for global optimization
        self._save_checkpoints = False
        if self.local_optimizer.writer is not None:
            self._tf_writer = tb.SummaryWriter(self.local_optimizer.writer.dir)
            self.update_writer(self.iteration)
            self.update_ckpt_saving()

    def callback(self, state, loss, accept):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The state vector of the local minimum found
        loss: float64
            The loss of the local minimum found
        accept: bool
            Whether of not the minimum was accepted
        """
        self.local_optimizer._iteration = 0
        if loss < self._best_minimum_f:
            self._best_minimum_f = loss
            self._best_minimum_iteration = self.iteration
            self._best_minimum_x = state
            if self.tf_writer is not None:
                self.tf_writer.add_scalar('Basin loss', loss, self.iteration)

            if self._save_checkpoints:
                time_passed = time.time() - self._ckpt_time
                if time_passed > self._ckpt_period:
                    self.local_optimizer.writer.save_ckpt_cbfn()
        else:
            shutil.rmtree(self.local_optimizer.writer.tf_writer.log_dir)

        self._iteration += 1
        if self.tf_writer is not None:
            self.update_writer(self.iteration)

    def update_writer(self, iteration):
        """
        Update summary writer to global iteration index.
        This method will place Basin<iteration> at the beginning of the path.

        Parameters
        ----------
        iteration: int
            The current iteration number.
        """
        log_dir = os.path.join(self.local_optimizer.writer.dir, 'Basin{}'.format(iteration))
        self.local_optimizer.writer._tf_writer = tb.SummaryWriter(log_dir)

    def update_ckpt_saving(self):
        """If checkpoint saving is defined then update to save only global minima."""
        cbfn_names = [cbfn.__name__ for cbfn in self.local_optimizer.writer.callback_fns]
        if 'save_ckpt_cbfn' in cbfn_names:
            self._save_checkpoints = True
            cbfn_index = cbfn_names.index('save_ckpt_cbfn')
            self._ckpt_period = self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_period']
            self._ckpt_time = self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_time']
            self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_period'] = np.Inf

    @property
    def method(self):
        return self._method

    @property
    def take_step(self):
        return self._take_step

    @property
    def local_minimizer_kwargs(self):
        return self._local_minimizer_kwargs

    @property
    def local_optimizer(self):
        return self._local_optimizer

    @property
    def iteration(self):
        return self._iteration

    @property
    def tf_writer(self):
        return self._tf_writer


class RandomStep(object):
    """"
    Replaces the default step taking routine of the basin hopping minimizer.
    The default step taking routine is a random displacement of the coordinates, but other step taking algorithms may be better for some systems.
    Here a custume step taking procedure is defined taking into account the parameter bounds.

    Parameters
    ----------
    medium: shdom.MediumEstimator
        A MediumEstimator object
    stepsize: float
        A factor to the per-parameter stepsize that is optimized throughout iterations

    Notes
    -----
    stepsize should be on the order of the scaled parameters (see preconditioning scale factor in GridDataEstimator)
    """
    def __init__(self, medium, stepsize=0.5):
        self.stepsize = stepsize
        bounds = medium.get_bounds()
        self.min_bound = [bound[0] for bound in bounds]
        self.max_bound = [bound[1] for bound in bounds]

    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize)
        return np.clip(x, self.min_bound, self.max_bound)
        
        
#--------------TESTs----------------------
## mediume:
#path = "/home/vhold/CloudCT/pyshdom/synthetic_cloud_fields/wiz/BOMEX_35x28x54_55080.txt"
## Generate a Microphysical medium
#droplets = shdom.MicrophysicalScatterer()
#droplets.load_from_csv('../synthetic_cloud_fields/jpl_les/rico32x37x26.txt', veff=0.1)
#droplets.load_from_csv(path)
#droplets.add_mie(self.medium.estimators['cloud'].mie)
#
## Rayleigh scattering for air molecules up to 20 km
#df = pd.read_csv('../ancillary_data/AFGL_summer_mid_lat.txt', comment='#', sep=' ')
#altitudes = df['Altitude(km)'].to_numpy(dtype=np.float32)
#temperatures = df['Temperature(k)'].to_numpy(dtype=np.float32)
#temperature_profile = shdom.GridData(shdom.Grid(z=altitudes), temperatures)
#air_grid = shdom.Grid(z=np.linspace(0, 100, 80))
#rayleigh = shdom.Rayleigh(wavelength=0.672)
#rayleigh.set_profile(temperature_profile.resample(air_grid))
#air = rayleigh.get_scatterer()
#atmospheric_grid = droplets.grid + air.grid # Add two grids by finding the common grid which maintains the higher resolution grid.
#atmosphere = shdom.Medium(atmospheric_grid)
#atmosphere.add_scatterer(droplets, name='cloud')
#atmosphere.add_scatterer(air, name='air')
##atmosphere.show_scatterer(name='cloud')
#
#
## ---------RTE SOLVE ----------------------------
#self.medium.set_state(state)
##self.medium.show_scatterer(name='cloud')
## -----------test below     
##self.medium.estimators['cloud'].lwc = atmosphere.scatterers['cloud'].lwc
##self.medium.estimators['cloud'].reff = atmosphere.scatterers['cloud'].reff
##self.medium.estimators['cloud'].veff = atmosphere.scatterers['cloud'].veff
##self.medium.estimators['cloud'].grid = atmosphere.scatterers['cloud'].grid
##self.medium.scatterers['air'] = atmosphere.scatterers['air']
## -----------test above  
#self.rte_solver.set_medium(self.medium)
##self.rte_solver.set_medium(atmosphere)
