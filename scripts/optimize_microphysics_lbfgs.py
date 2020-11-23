import os, time
import numpy as np
import shdom
import scipy.io as sio
import copy

from optimize_extinction_lbfgs import OptimizationScript as ExtinctionOptimizationScript


class OptimizationScript(ExtinctionOptimizationScript):
    """
    Optimize: Micro-physics
    ----------------------
    Estimate micro-physical properties based on multi-spectral radiance/polarization measurements.
    Note that for convergence a fine enough sampling of effective radii and variances should be pre-computed in the
    Mie tables used by the forward model. This is due to the linearization of the phase-function and it's derivatives.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_microphysics_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        super().__init__(scatterer_name)

    def medium_args(self, parser):
        """
        Add common medium arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--use_forward_lwc',
                            action='store_true',
                            help='Use the ground-truth LWC.')
        parser.add_argument('--use_forward_reff',
                                action='store_true',
                                help='Use the ground-truth effective radius.')
        parser.add_argument('--use_forward_veff',
                            action='store_true',
                            help='Use the ground-truth effective variance.')
        parser.add_argument('--const_lwc',
                            action='store_true',
                            help='Keep liquid water content constant at a specified value (not optimized).')
        parser.add_argument('--const_reff',
                            action='store_true',
                            help='Keep effective radius constant at a specified value (not optimized).')
        parser.add_argument('--const_veff',
                            action='store_true',
                            help='Keep effective variance constant at a specified value (not optimized).')
        parser.add_argument('--radiance_threshold',
                            default=[0.0175],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.'
                            'Threshold is either a scalar or a list of length of measurements.')
        parser.add_argument('--lwc_scaling',
                            default=10.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for liquid water content estimation')
        parser.add_argument('--reff_scaling',
                            default=1e-1,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective radius estimation')
        parser.add_argument('--veff_scaling',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective variance estimation')  
        parser.add_argument('--force_1d_reff',
                            action='store_true',
                            help='Use it if you want to retrieve the effective radius in 1D layers.')        
        parser.add_argument('--keep_curve',
                            default=0,
                            type=np.int32,
                            help='(default value: %(default)s) When keep_curve >0, there will be reff profile fitting every keep_curve iterations')          
        return parser

    def get_medium_estimator(self, measurements, ground_truth):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired measurements.
        ground_truth: shdom.Scatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """
        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            lwc_grid = ground_truth.lwc.grid
            if(self.args.force_1d_reff):
                # force 1D layerd grid of reff
                new_bounding_box = copy.deepcopy(ground_truth.reff.grid.bounding_box)
                reff_grid = shdom.Grid(bounding_box = new_bounding_box,
                                               z = ground_truth.lwc.grid.z)
            else:
                reff_grid = ground_truth.reff.grid
                
            veff_grid = ground_truth.reff.grid
        else:
            lwc_grid = reff_grid = veff_grid = self.cloud_generator.get_grid()
        grid = lwc_grid + reff_grid + veff_grid

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask = ground_truth.get_mask(threshold=0.01)
        else:
            carver = shdom.SpaceCarver(measurements)
            mask = carver.carve(grid, agreement=0.9, thresholds=self.args.radiance_threshold, PYTHON_SPACE_CURVE = self.args.python_space_curve)
            #---------------------------------------------------------------
            #--------CHACKE HERE IF THE MASK NOT TOO FAT FOR THE GRID-------
            #---------------------------------------------------------------
            egde_mask = np.zeros_like(mask.data)
            mask_shape = mask.data.shape
            egde_mask[0,:,:] = True
            egde_mask[mask_shape[0]-1,:,:] = True
            egde_mask[:,0,:] = True
            egde_mask[:,mask_shape[1]-1,:] = True
            egde_mask[:,:,0] = True
            egde_mask[:,:,mask_shape[2]-1] = True 
            A = np.bitwise_and(egde_mask , mask.data)
            MASK_TOO_FAT = np.any(A)
            if(MASK_TOO_FAT):
                raise Exception("The mask is too fat for the grid, you must pad the scatterer (to be optimized) with more zoros on the sides.")
            
            # Vadim added: Save the mask3d, just for the case we want to see how good we bound the cloudy voxels.
            if(self.args.save_gt_and_carver_masks):
                
                GT_mask = ground_truth.get_mask(threshold=0.01)
                sio.savemat(os.path.join(self.args.input_dir,'3D_masks.mat'), {'GT':GT_mask.data,'CARVER':mask.data,'thresholds':self.args.radiance_threshold})
                        
        # Define micro-physical parameters: either optimize, keep constant at a specified value or use ground-truth
        if self.args.use_forward_lwc:
            lwc = ground_truth.lwc
        elif self.args.const_lwc:
            lwc = self.cloud_generator.get_lwc(lwc_grid)
        else:
            if(isinstance(self.cloud_generator,shdom.generate.LesFile)):
                
                lwc = shdom.GridDataEstimator(self.cloud_generator.get_lwc(),
                                              min_bound=1e-5,
                                              max_bound=2.0,
                                              precondition_scale_factor=self.args.lwc_scaling)                
            else:
                
                if(self.cloud_generator.args.init == 'Monotonous'):
                    
                    lwc = shdom.GridDataEstimator(self.cloud_generator.get_lwc(grid=lwc_grid, min_lwc= 1.5e-5),
                                              min_bound=1e-5,
                                              max_bound=2.0,
                                              precondition_scale_factor=self.args.lwc_scaling)
                    
                else:
                    lwc = shdom.GridDataEstimator(self.cloud_generator.get_lwc(grid=lwc_grid),
                                              min_bound=1e-5,
                                              max_bound=2.0,
                                              precondition_scale_factor=self.args.lwc_scaling)
                    
                    
        lwc.apply_mask(mask)
        if not self.args.use_forward_mask:
            if(self.cloud_generator.args.init == 'Monotonous'):
                print('Here implement 20% reduction in the masked profile. ')

        if self.args.use_forward_reff:
            reff = ground_truth.reff
        elif self.args.const_reff:
            reff = self.cloud_generator.get_reff(reff_grid)
        else:
            if(isinstance(self.cloud_generator,shdom.generate.LesFile)):
                
                reff = shdom.GridDataEstimator(self.cloud_generator.get_reff(),
                                               min_bound=ground_truth.min_reff,
                                               max_bound=ground_truth.max_reff,
                                               precondition_scale_factor=self.args.reff_scaling)
                              
            else:
                
                if(self.cloud_generator.args.init == 'Monotonous'):
                
  
                    reff = shdom.GridDataEstimator(self.cloud_generator.get_reff(grid = reff_grid, min_reff = ground_truth.min_reff),
                                               min_bound=ground_truth.min_reff,
                                               max_bound=ground_truth.max_reff,
                                               precondition_scale_factor=self.args.reff_scaling)
        
                        
                else:
                    
                    reff = shdom.GridDataEstimator(self.cloud_generator.get_reff(grid = reff_grid),
                                               min_bound=ground_truth.min_reff,
                                               max_bound=ground_truth.max_reff,
                                               precondition_scale_factor=self.args.reff_scaling)

                    
                # still in not LesFile option:
                if(self.args.keep_curve > 0):
                    # if in evrey keep_curve iteration we must to fit theoretic curve:
                    def theoretic_data(x,z):
                        """
                        x - model parameters.
                        """
                        Z = z - x[2]
                        Z[Z<0] = 0
                        reff_theo = (x[0]*Z**(1./3.)) + x[1]
                        return reff_theo
                    
                    altitudes = reff_grid.z
                    # make the mask to fit only the profile:
                    profile_mask = np.mean(np.mean(mask.data, axis=0), axis=0)
                    profile_mask = profile_mask.astype('bool')
                    masked_altitudes = altitudes[profile_mask]
                    x0 = np.array([7,2.5,masked_altitudes.min()]) # initial model parameters.
                    profile = shdom.TheoreticProfile(parameters = 3, altitudes = masked_altitudes, data_fun = theoretic_data, every_iter = self.args.keep_curve)                        
                    profile.init_parameters(x0) 
                    reff.add_theoretic_profile(profile)
                    
                    
        reff.apply_mask(mask)
        if not self.args.use_forward_mask:
            if(self.cloud_generator.args.init == 'Monotonous'):
                print('Here implement 20% reduction in the masked profile. ')

        if self.args.use_forward_veff:
            veff = ground_truth.veff
        elif self.args.const_veff:
            veff = self.cloud_generator.get_veff(veff_grid)
        else:
            if(isinstance(self.cloud_generator,shdom.generate.LesFile)):
            
                veff = shdom.GridDataEstimator(self.cloud_generator.get_veff(),
                                               max_bound=ground_truth.max_veff,
                                               min_bound=ground_truth.min_veff,
                                               precondition_scale_factor=self.args.veff_scaling)
                
            else:
                
                
                veff = shdom.GridDataEstimator(self.cloud_generator.get_veff(veff_grid),
                                               max_bound=ground_truth.max_veff,
                                               min_bound=ground_truth.min_veff,
                                               precondition_scale_factor=self.args.veff_scaling)
        veff.apply_mask(mask)

        # Define a MicrophysicalScattererEstimator object
        cloud_estimator = shdom.MicrophysicalScattererEstimator(ground_truth.mie, lwc, reff, veff)
        cloud_estimator.set_mask(mask)

        # Create a medium estimator object (optional Rayleigh scattering)
        medium_estimator = shdom.MediumEstimator(
            loss_type=self.args.loss_type,
            stokes_weights=self.args.stokes_weights
        )
        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(cloud_estimator.wavelength)
            medium_estimator.set_grid(cloud_estimator.grid + air.grid)
            medium_estimator.add_scatterer(air, 'air')
        else:
            medium_estimator.set_grid(cloud_estimator.grid)
            
        medium_estimator.add_scatterer(cloud_estimator, self.scatterer_name)            

        return medium_estimator

    def load_forward_model(self, input_directory):
        """
        Load the ground-truth medium, rte_solver and measurements which define the forward model

        Parameters
        ----------
        input_directory: str
            The input directory where the forward model is saved

        Returns
        -------
        ground_truth: shdom.OpticalScatterer
            The ground truth scatterer
        rte_solver: shdom.RteSolverArray
            The rte solver with the numerical and scene parameters
        measurements: shdom.Measurements
            The acquired measurements
        """
        if hasattr(self, 'args'):
            if(self.args.cloudct_use):
                
                # Load forward model and cloud ct measurements
                medium, rte_solver, measurements = shdom.load_CloudCT_measurments_and_forward_model(input_directory)
                # if(isinstance(measurments,shdom.CloudCT_setup.SpaceMultiView_Measurements)
            else:
                
                # Load forward model and measurements
                medium, rte_solver, measurements = shdom.load_forward_model(input_directory)

        else:
            # Load forward model and cloud ct measurements
            medium, rte_solver, measurements = shdom.load_CloudCT_measurments_and_forward_model(input_directory)            
            
            
        # Get micro-physical medium ground-truth
        ground_truth = medium.get_scatterer(self.scatterer_name)

        
        return ground_truth, rte_solver, measurements

    #def get_ground_truth(self):
        #ground_truth, _, _ = self.load_forward_model(self.args.input_dir)
        ## here ground_truth is ???.
        #return ground_truth
    
    def get_summary_writer(self, measurements, ground_truth):
        """
        Define a SummaryWriter object

        Parameters
        ----------
        measurements: shdom.Measurements object
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground-truth scatterer for monitoring

        Returns
        -------
        writer: shdom.SummaryWriter object
            A logger for the TensorboardX.
        """
        writer = None
        if self.args.log is not None:
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.SummaryWriter(log_dir)
            writer.save_checkpoints(ckpt_period=20 * 60)
            writer.monitor_loss()

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.4, parameters=['lwc'])
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.2, parameters=['reff'])
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.01))

            # vadim added:
            writer.monitor_images(measurements=measurements, ckpt_period=5 * 60)
            
        return writer


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



