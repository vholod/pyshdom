import os, time
import numpy as np
import argparse

from scipy.optimize import fmin_l_bfgs_b

import shdom
import scipy.io as sio

class OptimizationScript(object):
    """
    Optimize: Extinction
    --------------------
    Estimate the extinction coefficient based on monochrome radiance measurements.
    In this script, the phase function, albedo and rayleigh scattering are assumed known and are not estimated.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_extinction_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name

    def optimization_args(self, parser):
        """
        Add common optimization arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--calc_only_cost',
                            action='store_true',
                            help='Use the calc_only_cost if you want only to calculate the cost of the initialized state.')
        parser.add_argument('--optimize_monotonous',
                            action='store_true',
                            help='Use the optimize_monotonous if you want find the optimal alpha reff and alpha lwc values for monotonous initialization.')
        parser.add_argument('--cloudct_use',
                            action='store_true',
                            help='Use it if you want to laoad and work with cloudct measurments.')          
        parser.add_argument('--python_space_curve',
                            action='store_true',
                            help='Use python implementation to calculate the 3d mask (space curving).')
        parser.add_argument('--input_dir',
                            help='Path to an input directory where the forward modeling parameters are be saved. \
                                  This directory will be used to save the optimization results and progress.')
        parser.add_argument('--reload_path',
                            help='Reload an optimizer or checkpoint and continue optimizing from that point.')
        parser.add_argument('--log',
                            help='Write intermediate TensorBoardX results. \
                                  The provided string is added as a comment to the specific run.')
        parser.add_argument('--save_gt_and_carver_masks',
                            action='store_true',
                            help='Use it if you want to save the curver m3d mask and the GT mask so you can compare them after the reconstruction.')                
        parser.add_argument('--use_forward_grid',
                            action='store_true',
                            help='Use the same grid for the reconstruction. This is a sort of inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--save_final3d',
                            action='store_true',
                            help='Use it if you want to save the retrieved parameters in 3D .mat file. It is \
                                  usefull for debugging/development.')        
        parser.add_argument('--use_forward_mask',
                            action='store_true',
                            help='Use the ground-truth cloud mask. This is an inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--add_noise',
                            action='store_true',
                            help='currently only supports AirMSPI noise model. \
                                  See shdom.AirMSPINoise object for more info.')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--globalopt',
                            action='store_true',
                            help='Global optimization with basin-hopping.'
                                 'For more info see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html')
        parser.add_argument('--maxiter',
                            default=1000,
                            type=int,
                            help='(default value: %(default)s) Maximum number of L-BFGS iterations.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--maxls',
                            default=30,
                            type=int,
                            help='(default value: %(default)s) Maximum number of line search steps (per iteration).'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--disp',
                            choices=[True, False],
                            default=True,
                            type=np.bool,
                            help='(default value: %(default)s) Display optimization progression.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--gtol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the maximum projected gradient.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--ftol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the relative change in loss function.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--stokes_weights',
                            nargs=4,
                            default=[1.0, 0.0, 0.0, 0.0],
                            type=float,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')
        parser.add_argument('--loss_type',
                            choices=['l2', 'normcorr'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
        parser.add_argument('--smoothness_const',
                            default=0,
                            type=np.float32,
                            help='Scale the regularization term for optimization. Currently only for the reff parameter.')
        
        
        return parser

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
        parser.add_argument('--use_forward_albedo',
                            action='store_true',
                            help='Use the ground truth albedo.')
        parser.add_argument('--use_forward_phase',
                            action='store_true',
                            help='Use the ground-truth phase reconstruction.')
        parser.add_argument('--radiance_threshold',
                            default=[0.05],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.' \
                            'Threshold is either a scalar or a list of length of measurements.')
        parser.add_argument('--mie_base_path',
                            default='mie_tables/polydisperse/Water_<wavelength>nm.scat',
                            help='(default value: %(default)s) Mie table base file name. ' \
                                 '<wavelength> will be replaced by the corresponding wavelength.')

        return parser

    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.

        Returns
        -------
        args: arguments from argparse.ArgumentParser()
            Arguments required for this script.
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        parser = argparse.ArgumentParser()
        parser = self.optimization_args(parser)
        parser = self.medium_args(parser)

        # Additional arguments to the parser
        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--init')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--init',
                            default='Homogeneous',
                            help='(default value: %(default)s) Name of the generator used to initialize the atmosphere. \
                                  for additional generator arguments: python scripts/optimize_extinction_lbgfs.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        init = subparser.parse_known_args()[0].init
        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh

        CloudGenerator = None
        if init:
            CloudGenerator = getattr(shdom.generate, init)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None

    def get_medium_estimator(self, measurements, ground_truth):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground truth scatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """
        wavelength = ground_truth.wavelength

        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            extinction_grid = ground_truth.extinction.grid
            albedo_grid = ground_truth.albedo.grid
            phase_grid = ground_truth.phase.grid
        else:
            extinction_grid = albedo_grid = phase_grid = self.cloud_generator.get_grid()
        grid = extinction_grid + albedo_grid + phase_grid

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask = ground_truth.get_mask(threshold=1.0)
        else:
            carver = shdom.SpaceCarver(measurements)
            
            mask = carver.carve(grid, agreement=0.9, thresholds=self.args.radiance_threshold, PYTHON_SPACE_CURVE = self.args.python_space_curve)
            
            # Vadim added: Save the mask3d, just for the case we want to see how good we bound the cloudy voxels.
            if(self.args.save_gt_and_carver_masks):
                
                GT_mask = ground_truth.get_mask(threshold=1.0)
                sio.savemat(os.path.join(self.args.input_dir,'3D_masks.mat'), {'GT':GT_mask.data,'CARVER':mask.data,'thresholds':self.args.radiance_threshold})
                        
        # Define the known albedo and phase: either ground-truth or specified, but it is not optimized.
        if self.args.use_forward_albedo is False or self.args.use_forward_phase is False:
            table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength)))
            self.cloud_generator.add_mie(table_path)

        if self.args.use_forward_albedo:
            albedo = ground_truth.albedo
        else:
            albedo = self.cloud_generator.get_albedo(wavelength, albedo_grid)

        if self.args.use_forward_phase:
            phase = ground_truth.phase
        else:
            phase = self.cloud_generator.get_phase(wavelength, phase_grid)

        extinction = shdom.GridDataEstimator(self.cloud_generator.get_extinction(grid=grid),
                                             min_bound=1e-3,
                                             max_bound=2e2)
        cloud_estimator = shdom.OpticalScattererEstimator(wavelength, extinction, albedo, phase)
        cloud_estimator.set_mask(mask)

        # Create a medium estimator object (optional Rayleigh scattering)
        medium_estimator = shdom.MediumEstimator()
        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(wavelength)
            medium_estimator.set_grid(cloud_estimator.grid + air.grid)
                        
            medium_estimator.add_scatterer(air, 'air')
        else:
            medium_estimator.set_grid(cloud_estimator.grid)

        medium_estimator.add_scatterer(cloud_estimator, self.scatterer_name)
        
        return medium_estimator

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
            writer.monitor_shdom_iterations()
            writer.monitor_images(measurements=measurements, ckpt_period=5 * 60)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.4)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=1.0))

        return writer

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
        if(self.args.cloudct_use):
            
            # Load forward model and cloud ct measurements
            medium, rte_solver, measurements = shdom.load_CloudCT_measurments_and_forward_model(input_directory)
            # if(isinstance(measurments,shdom.CloudCT_setup.SpaceMultiView_Measurements)
        else:
            
            # Load forward model and measurements
            medium, rte_solver, measurements = shdom.load_forward_model(input_directory)
        
        # Get optical medium ground-truth
        ground_truth = medium.get_scatterer(self.scatterer_name)
        if isinstance(ground_truth, shdom.MicrophysicalScatterer):
            ground_truth = ground_truth.get_optical_scatterer(measurements.wavelength)
        return ground_truth, rte_solver, measurements

    
    def get_ground_truth(self):
        ground_truth, _, _ = self.load_forward_model(self.args.input_dir)
        # here ground_truth is shdom.medium.OpticalScatterer.
        return ground_truth
    
    def get_optimizer(self):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """
        

        ground_truth, rte_solver, measurements = self.load_forward_model(self.args.input_dir)

        # Add noise (currently only supports AirMSPI noise model)
        if self.args.add_noise:
            noise = shdom.AirMSPINoise()
            measurements = noise.apply(measurements)

        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements, ground_truth)

        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements, ground_truth)

        # Initialize a LocalOptimizer
        options = {
            'maxiter': self.args.maxiter,
            'maxls': self.args.maxls,
            'disp': self.args.disp,
            'gtol': self.args.gtol,
            'ftol': self.args.ftol
        }
        optimizer = shdom.LocalOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs, smoothness_ratio = self.args.smoothness_const)
        optimizer.set_measurements(measurements)
        optimizer.set_rte_solver(rte_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_writer(writer)

        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)
        return optimizer

    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()
        
        if(self.args.calc_only_cost):
            # Here we just calculate the cost of the initialized state.
            # No need of the optimization.
            ground_truth, rte_solver, measurements = self.load_forward_model(self.args.input_dir)
            # Initialize a Medium Estimator
            medium_estimator = self.get_medium_estimator(measurements, ground_truth)
            
            # prepere the solvers:
            if not isinstance(rte_solver, shdom.RteSolverArray):
                rte_solver = shdom.RteSolverArray([rte_solver])
            for solver in rte_solver:
                solver.set_medium(medium_estimator)  
                
            # solve rte:
            rte_solver.solve(maxiter=100, verbose=False)
                
            cost = medium_estimator.compute_cost(rte_solver, measurements, self.args.n_jobs) 
            
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log)
            tracker_path = os.path.join(log_dir,'cost.txt')
            with open(tracker_path, 'a') as file_object:
                file_object.write('{}\n'.format(cost))

        elif self.args.optimize_monotonous:


            self.reff_scale = 0.1
            self.lwc_scale = 10.0
            self.reff_pow = 1.0 / 3.0
            self.lwc_pow = 1.0
            self.medium_mono, rte_solver, self.measurements_mono = self.load_forward_model(self.args.input_dir)

            GT_mask = self.medium_mono.get_mask(threshold=0.01)
            z = GT_mask.grid.z
            # Find the bottom index and point of mask:
            cumsum_volume_axis2 = np.cumsum(GT_mask.data, axis=2, dtype=float)
            tmp = np.sum(np.sum(cumsum_volume_axis2, axis=1), axis=0)
            tmp[tmp == 0] = max(tmp)
            # GET RID OF ZEROS -> done
            Bottom = np.argmin(tmp)
            self.z0 = z[Bottom]
            print("Cloud base at z0={}km".format(self.z0))
            self.results = {'time':[], 'init_lwc':[], 'init_reff':[],'lwc':[], 'reff':[], 'loss':[]}
            # prepare the solvers:
            if not isinstance(rte_solver, shdom.RteSolverArray):
                self.rte_solver_mono = shdom.RteSolverArray([rte_solver])
            else:
                self.rte_solver_mono = rte_solver
            self.iter_ = 0
            self.optimizer_ = shdom.LocalOptimizer('L-BFGS-B', n_jobs=self.args.n_jobs)
            def call_back_func(x):
                print(f'iter:{self.iter_} lwc: {round(x[0] / self.lwc_scale, 4)}, reff:{round(x[1] / self.reff_scale, 4)}')
                self.iter_+=1

             # TODO
            """ 
            import matplotlib.pyplot as plt
            x={'time': [111.31845927238464, 111.29790425300598, 85.52190637588501, 115.99412488937378, 107.25484251976013, 114.94710731506348, 109.5345447063446, 111.49884366989136, 113.71950030326843, 111.5090880393982, 106.08210802078247, 141.2928819656372, 109.33372235298157, 98.77369832992554, 114.2970700263977, 107.49797534942627, 105.40637254714966, 103.88471722602844, 108.58827424049377, 115.96372652053833], 'init_lwc': [0.004145161260341412, 0.17834323255049983, 0.14229991201641212, 0.0729905138895786, 0.11764251053993446, 0.04774122054376628, 0.11558167598065656, 0.06170146622227331, 0.025665230843357124, 0.0398725816785728, 0.19332319208922577, 0.10233001368147937, 0.10496643375146104, 0.11793777214074067, 0.11209991406063546, 0.17353426238530278, 0.060742032980532205, 0.040861206639375675, 0.12745868158152607, 0.19525907552451388], 'init_reff': [10.021833499735703, 16.7878207373198, 1.8011839004906505, 8.409714520183098, 18.662850464419783, 5.458243415252058, 16.39715385802458, 16.453715752016944, 6.187033784125169, 19.048887526459122, 19.789071104191148, 11.073132558676415, 17.119237339859534, 5.3357056183797695, 9.384763375488186, 4.990524692022628, 17.988444623613734, 12.023916521153366, 19.76336970540251, 14.202736253115884], 'lwc': [0.6676204154994683, 1.0377264859941617, 0.24229990774658045, 0.540198120449134, 1.1135798322489932, 0.4018041630589598, 1.0045984914778452, 0.9933219140006878, 0.4300082909896969, 1.104732768476901, 1.1873780176650233, 0.6876856134810152, 1.0366496037516144, 0.4055956717920119, 0.5940106455623853, 0.3755800265604726, 1.0643669708818635, 0.7742006299630354, 1.1687087378168806, 0.8309181146427764], 'reff': [9.992791904320818, 16.745957979635673, 1.7982616302854877, 8.391302548348975, 18.615799478345735, 5.445542969618444, 16.355065612541928, 16.41097074823329, 6.1724569785076016, 19.001015261936022, 19.740366060710684, 11.047559225806811, 17.075411525198884, 5.325165827259237, 9.364937275155025, 4.983527486681762, 17.94259063961043, 11.990502012803345, 19.713921172899518, 14.17481504347776], 'loss': [0.18708064468949542, 0.19359235396001784, 0.2278375584476526, 0.1812080130877353, 0.19174426764572552, 0.18464762888850061, 0.19108404377856525, 0.18913693526948683, 0.18283245446619514, 0.19000397488465737, 0.1946182490683079, 0.18215427428017422, 0.1907580622877526, 0.1870307759700128, 0.18111221382669304, 0.18537871907678224, 0.18976628889252734, 0.18787102646662793, 0.1925530319179792, 0.1846800880937834]}

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x['reff'], x['lwc'], x['loss'])
        Backend Qt5Agg is interactive backend. Turning interactive mode on.
        Out[3]: <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x7f07d50ae450>
"""
            # best_parameters, values, experiment, model = optimize(
            #     parameters=[
            #         {
            #             "name": "init_lwc",
            #             "type": "range",
            #             "bounds": [0.0, 20.0],
            #             "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            #             "log_scale": False,  # Optional, defaults to False.
            #         },
            #         {
            #             "name": "init_reff",
            #             "type": "range",
            #             "bounds": [0.0, 20.0],
            #         }
            #     ],
            #     experiment_name="test",
            #     objective_name="fmin_l_bfgs_b",
            #     evaluation_function=monotonous_optimization_func_call,
            #     minimize=True,  # Optional, defaults to False.
            #     #parameter_constraints=["x1 + x2 <= 20"],  # Optional.
            #     #outcome_constraints=["l2norm <= 1.25"],  # Optional.
            #     total_trials=5,  # Optional.
            # )
            # def monotonous_optimization_func_call(params):
            #     init_lwc = params.get('init_lwc')
            #     init_reff = params.get('init_reff')
            #     result = fmin_l_bfgs_b(func=self.optimize_monotonous_objective_func,
            #                            x0=np.array([init_lwc * self.lwc_scale, init_reff * self.reff_scale]),
            #                            bounds=[(0, float('inf')), (0, float('inf'))],
            #                            callback=call_back_func,
            #                            maxls=5,
            #                            factr=10.0)

            import time
            for _ in range(20):
                start = time.time()
                init_reff = np.random.random() * 20  # 13.0
                init_lwc = np.random.random() * 0.2  # 1.0
                result = fmin_l_bfgs_b(func=self.optimize_monotonous_objective_func,
                                       x0=np.array([init_lwc*self.lwc_scale, init_reff*self.reff_scale]),
                                       bounds=[(0, float('inf')), (0, float('inf'))],
                                       callback=call_back_func,
                                       maxls=5,
                                       factr=10.0)
                end = time.time()

                print(result)
                self.results['time'].append(end - start)
                self.results['init_lwc'].append(init_lwc)
                self.results['init_reff'].append(init_reff)
                self.results['lwc'].append(result[0][0]/self.lwc_scale)
                self.results['reff'].append(result[0][1]/self.reff_scale)
                self.results['loss'].append(result[1])
            print(self.results)
                
        else:
            
            # perform full optimization:
            local_optimizer = self.get_optimizer()
    
            # Optimization process
            num_global_iter = 1
            if self.args.globalopt:
                global_optimizer = shdom.GlobalOptimizer(local_optimizer = local_optimizer)
                result = global_optimizer.minimize(niter_success=20, T=1e-3)
                num_global_iter = result.nit
                result = result.lowest_optimization_result
                local_optimizer.set_state(result.x)
            else:
                result = local_optimizer.minimize()
    
            print('\n------------------ Optimization Finished ------------------\n')
            print('Number global iterations: {}'.format(num_global_iter))
            print('Success: {}'.format(result.success))
            print('Message: {}'.format(result.message))
            print('Final loss: {}'.format(result.fun))
            print('Number iterations: {}'.format(result.nit))
    
            # Save optimizer state
            save_dir = local_optimizer.writer.dir if self.args.log is not None else self.args.input_dir
            local_optimizer.save_state(os.path.join(save_dir, 'final_state.ckpt'))
    
            # vadim added to save final 3D reconstructions:
            if(self.args.save_final3d):  
                ground_truth = self.get_ground_truth()
                #if not isinstance(ground_truth, shdom.MicrophysicalScatterer):
                    ## If we here, it means that the optimization was called on extinction and not for microphysics.
                    #GT_parameter = ground_truth.extinction.data = ground_truth.extinction.data
                    
                rec_scatterer = local_optimizer.medium.get_scatterer(name=self.scatterer_name)
                # local_optimizer.medium is shdom.optimize.MediumEstimator object.
                for parameter_name, parameter in rec_scatterer.estimators.items():
                    rec_param = parameter.data 
                    try:
                        dx = parameter.grid.dx
                        dy = parameter.grid.dy
                    except AttributeError:
                        dx = -1
                        dy = -1
                        
                    nz = parameter.grid.nz
                    dz = (parameter.grid.zmax - parameter.grid.zmin)/nz
                    GT_parameter = ground_truth.__getattribute__(parameter_name).data
                    sio.savemat(os.path.join(save_dir,'FINAL_3D_{}.mat'.format(parameter_name)), {'GT':GT_parameter,parameter_name:rec_param,'dx':dx,'dy':dy,'dz':dz})
                
    def optimize_monotonous_objective_func(self, x):
        # Initialize a Medium Estimator

        self.args.lwc = x[0]/self.lwc_scale
        self.args.reff = x[1]/self.reff_scale
        medium_estimator = self.get_medium_estimator(self.measurements_mono, self.medium_mono)
        # medium_estimator.
        for solver in self.rte_solver_mono:
            solver.set_medium(medium_estimator)

        # solve rte:
        self.rte_solver_mono.solve(maxiter=2, verbose=False) # TODO is this necessary?
        self.optimizer_.set_measurements(self.measurements_mono)
        self.optimizer_.set_rte_solver(self.rte_solver_mono)
        self.optimizer_.set_medium_estimator(medium_estimator)
        self.optimizer_.init_optimizer()

        _, loss, _, grad_3d = self.optimizer_._medium.compute_gradient(self.rte_solver_mono, self.measurements_mono, self.args.n_jobs, iteration=1)
        grid = self.medium_mono.lwc.grid # can be any param grid
        _, _, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
        z_diff = np.abs(Z-self.z0)
        lwc_grad = z_diff**self.lwc_pow
        reff_grad = z_diff**self.reff_pow
        grad = np.empty(2)
        grad[0] = (grad_3d[0]*lwc_grad).sum()*self.lwc_scale
        grad[1] = (grad_3d[1]*reff_grad).sum()*self.reff_scale
        print(f'At lwc: {round(x[0]/self.lwc_scale,4)}, reff:{round(x[1]/self.reff_scale,4)}: Loss:{round(loss,6)}, grad:{np.round(grad,4)}')
        return loss, grad

if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()




