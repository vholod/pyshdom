import numpy as np
import argparse
import shdom


class RenderScript(object):
    """
    Render: Radiance for perspective camera
    -----------------------------------------------
    Forward rendering of an atmospheric medium with at multiple spectral bands with an perspective sensor measuring
    
    As with all `render` scripts a `generator` needs to be specified using the generator flag (--generator).
    The Generator defines the medium parameters: Grid, Extinction, Single Scattering Albedo and Phase function with it's own set of command-line flags.

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/render/render_radiance_toa.py --help

    For a tutorial overview of how to operate the forward rendering model see the following notebooks:
     - notebooks/Make Mie Table.ipynb
     - notebooks/Radiance Rendering [Single Image].ipynb
     - notebooks/Radiance Rendering [Multiview].ipynb
     - notebooks/Radiance Rendering [Multispectral].ipynb

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name
        self.num_stokes = 1
        self.sensor = shdom.RadianceSensor()

    def solution_args(self, parser):
        """
        Add common RTE solution arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('output_dir',
                            help='Path to an output directory where the measurements and model parameters will be saved. \
                                  If the folder doesnt exist, it will be created.')
        parser.add_argument('wavelength',
                            nargs='+',
                            type=np.float32,
                            help='Wavelengths for the measurements [micron].')
        parser.add_argument('--solar_zenith',
                            default=165.0,
                            type=np.float32,
                            help='(default value: %(default)s) Solar zenith [deg]. This is the direction of the photons in range (90, 180]')
        parser.add_argument('--solar_azimuth',
                            default=0.0,
                            type=np.float32,
                            help='(default value: %(default)s) Solar azimuth [deg]. This is the direction of the photons')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--surface_albedo',
                            default=0.05,
                            type=float,
                            help='(default value: %(default)s) The albedo of the Lambertian Surface.')
        parser.add_argument('--num_mu',
                            default=16,
                            type=int,
                            help='(default value: %(default)s) The number of discrete ordinates in the zenith direction.\
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--num_phi',
                            default=32,
                            type=int,
                            help='(default value: %(default)s) The number of discrete ordinates in the azimuthal direction.\
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--split_accuracy',
                            default=0.03,
                            type=float,
                            help='(default value: %(default)s) The cell splitting accuracy for SHDOM. \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--solution_accuracy',
                            default=1e-4,
                            type=float,
                            help='(default value: %(default)s) The SHDOM solution criteria (accuracy). \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--adapt_grid_factor',
                            default=10,
                            type=float,
                            help='(default value: %(default)s) The radio of adaptive (internal) grid points property array grid points \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--solar_spectrum',
                            action='store_true',
                            help='Use solar spectrum flux for each wavelength. \
                            If not used, the solar flux is normalized to be 1.0. \
                            See shdom.SolarSpectrum for more details.')
        return parser

    def rendering_args(self, parser):
        """
        Add common rendering arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--cam_x',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera position in x axis [km] (North)')
        parser.add_argument('--cam_y',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera position in y axis [km] (East)')
        parser.add_argument('--cam_z',
                            default=[100.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera position in z axis [km] (up)')
        
        parser.add_argument('--lookat_x',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera look at position in x axis [km] (North)')
        parser.add_argument('--lookat_y',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera look at position in y axis [km] (East)')
        parser.add_argument('--lookat_z',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Camera look at position in z axis [km] (up)')
        
        parser.add_argument('--fov',
                            default=30.0,
                            type=np.float32,
                            help='(default value: %(default)s) Camera field of view in [degree]. It is one fov for all cameras.')
        
        parser.add_argument('--cam_nx',
                            default=30.0,
                            type=int,
                            help='(default value: %(default)s) Number of pixels in camera x axis (North). It is fixed for all cameras.')


        parser.add_argument('--cam_ny',
                            default=30.0,
                            type=int,
                            help='(default value: %(default)s) Number of pixels in camera y axis (East). It is fixed for all cameras.')
        
        parser.add_argument('--up_x',
                            default=0.0,
                            type=np.float32,
                            help='(default value: %(default)s) Up vector x axis . It is fixed for all cameras.')
        
        parser.add_argument('--up_y',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Up vector y axis . It is fixed for all cameras.')
        
        parser.add_argument('--up_z',
                            default=0.0,
                            type=np.float32,
                            help='(default value: %(default)s) Up vector z axis . It is fixed for all cameras.')        
                
        parser.add_argument('--mie_base_path',
                            default='mie_tables/polydisperse/Water_<wavelength>nm.scat',
                            help='(default value: %(default)s) Mie table base file name. '\
                                 '<wavelength> will be replaced by the corresponding wavelengths.')
        return parser

    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.
        """
        parser = argparse.ArgumentParser()
        parser = self.solution_args(parser)
        parser = self.rendering_args(parser)

        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--generator')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--generator',
                            help='Name of the generator used to generate the atmosphere. \
                                  or additional generator arguments: python scripts/render_radiance_toa.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh
        generator = subparser.parse_known_args()[0].generator

        CloudGenerator = None
        if generator:
            CloudGenerator = getattr(shdom.Generate, generator)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.Generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator, self.air_generator = self.init_generators(CloudGenerator, AirGenerator)

    def init_generators(self, CloudGenerator, AirGenerator):
        """
        Initialize the medium generators. The cloud generator also loads the Mie scattering
        tables for the given wavelengths at this point.

        Parameters
        -------
        CloudGenerator: a shdom.Generator class object.
            Creates the cloudy medium.
        AirGenerator: a shdom.Air class object
            Creates the scattering due to air molecules

        Returns
        -------
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        cloud_generator = CloudGenerator(self.args)
        for wavelength in self.args.wavelength:
            table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength)))
            cloud_generator.add_mie(table_path)

        air_generator = None
        if self.args.add_rayleigh:
            air_generator = AirGenerator(self.args)
        return cloud_generator, air_generator

    def get_medium(self):
        """
        Generate an atmospheric domain

        returns
        -------
        atmosphere: shdom.Medium object.
            Creates the atmospheric medium.
        """
        cloud = self.cloud_generator.get_scatterer()
        atmosphere = shdom.Medium()
        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(cloud.wavelength)
            atmosphere.set_grid(cloud.grid + air.grid)
            atmosphere.add_scatterer(air, 'air')
        else:
            atmosphere.set_grid(cloud.grid)
        atmosphere.add_scatterer(cloud, self.scatterer_name)
        return atmosphere

    def get_solver(self, medium):
        """
        Define an RteSolverArray object

        Parameters
        ----------
        medium: shdom.Medium
            The atmospheric Medium for which the RTE is solved

        Returns
        -------
        rte_solver: shdom.RteSolverArray object
            A solver array initialized to the input medium and numerical and scene arguments
        """
        rte_solvers = shdom.RteSolverArray()
        if self.args.solar_spectrum:
            solar_spectrum = shdom.SolarSpectrum()
            solar_fluxes = solar_spectrum.get_monochrome_solar_flux(self.args.wavelength)
            solar_fluxes = solar_fluxes / max(solar_fluxes)
        else:
            solar_fluxes = np.full_like(self.args.wavelength, 1.0)

        numerical_params = shdom.NumericalParameters(
            num_mu_bins=self.args.num_mu,
            num_phi_bins=self.args.num_phi,
            split_accuracy=self.args.split_accuracy,
            adapt_grid_factor=self.args.adapt_grid_factor,
            solution_accuracy=self.args.solution_accuracy
        )
        for wavelength, solar_flux in zip(self.args.wavelength, solar_fluxes):
            scene_params = shdom.SceneParameters(
                wavelength=wavelength,
                source=shdom.SolarSource(self.args.solar_azimuth, self.args.solar_zenith, solar_flux),
                surface=shdom.LambertianSurface(albedo=self.args.surface_albedo)
            )
            rte_solver = shdom.RteSolver(scene_params, numerical_params, num_stokes=self.num_stokes)
            rte_solver.set_medium(medium)
            rte_solvers.add_solver(rte_solver)
        return rte_solvers

    def render(self, bounding_box, rte_solver):
        """
        Define a sensor and render a perspective image at the top domain.

        Parameters
        ----------
        bounding_box: shdom.BoundingBox object
            Used to compute the projection that will see the entire bounding box.
        rte_solver: shdom.RteSolverArray
            A solver that contains the solution to the RTE in the medium

        Returns
        -------
        measurements: shdom.Measurements object
            Encapsulates the measurements and sensor geometry for later optimization
        """
        projection = shdom.MultiViewProjection()
        
        assert len(self.args.cam_x) == len(self.args.cam_y) == len(self.args.cam_z), "Some camera parameters are missing (camera location)"
        assert len(self.args.lookat_x) == len(self.args.lookat_y) == len(self.args.lookat_z), "Some camera parameters are missing (camera lookat)"
        if(not (len(self.args.lookat_x) == len(self.args.cam_x))):
            if(len(self.args.lookat_x) == 1):
                # duplicate lookat vectors so it will have length as the number of cameras
                self.args.lookat_x = self.args.lookat_x * len(self.args.cam_x)
                self.args.lookat_y = self.args.lookat_y * len(self.args.cam_x)
                self.args.lookat_z = self.args.lookat_z * len(self.args.cam_x)
                
            else:
                raise Exception('Camera lookat parameters are missing')
           
        origin_list = [list(i) for i in zip(self.args.cam_x, self.args.cam_y, self.args.cam_z)]
        lookat_list = [list(i) for i in zip(self.args.lookat_x, self.args.lookat_y, self.args.lookat_z)]
        up = [self.args.up_x, self.args.up_y, self.args.up_z]
        
        fov = self.args.fov
        nx = int(self.args.cam_nx)
        ny = int(self.args.cam_ny)
        
        for origin, lookat in zip(origin_list, lookat_list):
            x, y, z = origin
            tmpproj = shdom.PerspectiveProjection(fov, nx, ny, x, y, z)
            tmpproj.look_at_transform(lookat,up)
            projection.add_projection(tmpproj)
            
            
        camera = shdom.Camera(self.sensor, projection)
        images = camera.render(rte_solver, self.args.n_jobs)
        measurements = shdom.Measurements(camera, images=images, wavelength=rte_solver.wavelength)
        return measurements

    def SolveAndRender(self):
        
        self.parse_arguments()        
        medium = self.get_medium()
        rte_solver = self.get_solver(medium)
        rte_solver.solve(maxiter=100,init_solution=False, verbose=True)
        measurements = self.render(medium.get_scatterer('cloud').bounding_box, rte_solver)
        # bring it back to main.
        return rte_solver, measurements, medium
        
    def main(self):
        """
        Main forward rendering script.
        """
        rte_solver, measurements, medium = self.SolveAndRender()
        # Save measurements, medium and solver parameters
        shdom.save_forward_model(self.args.output_dir, medium, rte_solver, measurements)


if __name__ == "__main__":
    script = RenderScript(scatterer_name='cloud')
    script.main()
    print('Finished to render all')
