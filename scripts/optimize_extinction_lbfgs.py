import argparse
import os
import time

import numpy as np
import scipy.io as sio

import shdom


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
        parser.add_argument('--cloudct_use',
                            action='store_true',
                            help='Use it if you want to laoad and work with cloudct measurments.')
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
        parser.add_argument('--net_initialization_path',
                            help='Path to the mat file of the initialization.')
        parser.add_argument('--net_initialize_solution',
                            action='store_true',
                            help='If to initialize solution.')
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

            mask = carver.carve(grid, agreement=0.9, thresholds=self.args.radiance_threshold)

            # Vadim added: Save the mask3d, just for the case we want to see how good we bound the cloudy voxels.
            if (self.args.save_gt_and_carver_masks):
                GT_mask = ground_truth.get_mask(threshold=1.0)
                sio.savemat(os.path.join(self.args.input_dir, '3D_masks.mat'),
                            {'GT': GT_mask.data, 'CARVER': mask.data, 'thresholds': self.args.radiance_threshold})

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

        extinction = self.cloud_generator.get_extinction(grid=grid)
        if self.args.net_initialize_solution:
            extinction._data = sio.loadmat(self.args.net_initialization_path)['extinction']
        extinction = shdom.GridDataEstimator(extinction, min_bound=1e-3, max_bound=2e2)
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
            log_dir = os.path.join(self.args.input_dir, 'logs',
                                   self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.SummaryWriter(log_dir)
            writer.save_checkpoints(ckpt_period=20 * 60)
            writer.monitor_loss()
            writer.monitor_shdom_iterations()
            writer.monitor_images(measurements=measurements, ckpt_period=5 * 60)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth,
                                        dilute_percent=0.4)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth,
                                           ground_truth_mask=ground_truth.get_mask(threshold=1.0))

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
        if (self.args.cloudct_use):

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
        self.parse_arguments()

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
        optimizer = shdom.LocalOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs)
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
        local_optimizer = self.get_optimizer()

        # Optimization process
        num_global_iter = 1
        if self.args.globalopt:
            global_optimizer = shdom.GlobalOptimizer(local_optimizer=local_optimizer)
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
        if (self.args.save_final3d):
            ground_truth = self.get_ground_truth()
            # if not isinstance(ground_truth, shdom.MicrophysicalScatterer):
            ## If we here, it means that the optimization was called on extinction and not for microphysics.
            # GT_parameter = ground_truth.extinction.data = ground_truth.extinction.data

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
                dz = (parameter.grid.zmax - parameter.grid.zmin) / nz
                GT_parameter = ground_truth.__getattribute__(parameter_name).data
                sio.savemat(os.path.join(save_dir, 'FINAL_3D_{}.mat'.format(parameter_name)),
                            {'GT': GT_parameter, parameter_name: rec_param, 'dx': dx, 'dy': dy, 'dz': dz})


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()




