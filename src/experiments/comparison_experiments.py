import numpy as np
import params
import warnings
from typing import Dict

from src.sql.entity.bulk_collector import BulkCollector
from src.graphics import Graphics, nanmean
from src.progress import StatusHandler
from src.sql.connection import Connection
from src.sql.entity.simulation import load_simulation, Simulation


class ComparisonExperiments(Graphics):
    def __init__(self, output_name: str, projects: [str], save_figures: bool):
        self.projects = projects
        self.output_name = output_name

        self.status = StatusHandler()

        self.status.print('Connecting to database...')
        self.connection = Connection(params.DATABASE_NAME)
        self.status.println('Connection successful')

        self.status.print('Loading simulations...')
        self.data: Dict[str, Simulation] = {project: load_simulation(self.connection, project, self.status) for project in projects}
        self.status.println('Simulations loaded')

        self.status.print('Loading meta data of simulations...')
        self.hdf5_iterations = {project: sorted(list(self.data[project].hdf5_iterations.keys())) for project in projects}
        self.x_ticks_hdf5 = self.get_x_ticks(self.hdf5_iterations)
        self.status.println('Meta data loaded')

        # Init Graphics
        super().__init__(output_name, save_figures)

    @staticmethod
    def np_nan_mean(a: np.ndarray, axis=None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(a, axis)

    def get_x_ticks(self, iterations: Dict[str, list]) -> Dict[str, np.ndarray]:
        return {
            project: np.around(np.array(iterations[project]) * self.data[project].config.dt / 1e-3, decimals=1)
            for project in self.projects
        }

    def velocity_analysis(self, end_iteration: int, n_iterations_before: int = 1):
        for project in self.projects:
            self.set_x_ticks(self.x_ticks_hdf5[project])

            self.status.print(f"Loading velocity data of {project}")
            iterations = list(filter(lambda i: i <= end_iteration, self.hdf5_iterations[project]))[-n_iterations_before:]

            bulk_load = BulkCollector(
                self.connection,
                self.data[project].id,
                {i: self.data[project].hdf5_iterations[i] for i in iterations},
                {},
                self.status
            )

            velocity_data = [row.exclude_borders(self.data[project].boundary) for row in bulk_load.fluid_velocities()]
            self.status.println(f"Loaded velocity data of {project}")

            velocity = nanmean([data.magnitude for data in velocity_data], axis=0) * 1e6
            velocity_y = np.absolute(nanmean([data.y for data in velocity_data], axis=0)) * 1e6

            for data, data_type in [
                (velocity, 'velocity'),
                (velocity_y, 'velocity-y')
            ]:
                self.plot_vertical_slice(data, project, data_type)
                self.plot_horizontal_slice(data, project, data_type)
                self.plot_moving_average_over_height(data, 5, project, data_type)
                self.plot_moving_average_over_height(data, 9, project, data_type)
                self.plot_moving_average_over_height(data, 19, project, data_type)
                self.plot_moving_average_over_width(data, 5, project, data_type)
                self.plot_moving_average_over_width(data, 9, project, data_type)
                self.plot_moving_average_over_width(data, 19, project, data_type)

    def shear_stress_analysis(self, end_iteration: int, n_iterations_before: int = 1):
        for project in self.projects:
            self.set_x_ticks(self.x_ticks_hdf5[project])

            self.status.print(f"Loading shear stress data of {project}")
            iterations = list(filter(lambda i: i <= end_iteration, self.hdf5_iterations[project]))[-n_iterations_before:]

            bulk_load = BulkCollector(
                self.connection,
                self.data[project].id,
                {i: self.data[project].hdf5_iterations[i] for i in iterations},
                {},
                self.status
            )

            shear_stress_data = [row.exclude_borders(self.data[project].boundary) for row in bulk_load.fluid_shear_stress()]
            self.status.println(f"Loaded shear stress data of {project}")

            data = nanmean([data.magnitude for data in shear_stress_data], axis=0)
            data_type = 'shear stress'

            self.plot_vertical_slice(data, project, data_type)
            self.plot_horizontal_slice(data, project, data_type)
            self.plot_moving_average_over_height(data, 5, project, data_type)
            self.plot_moving_average_over_height(data, 9, project, data_type)
            self.plot_moving_average_over_height(data, 19, project, data_type)
            self.plot_moving_average_over_width(data, 5, project, data_type)
            self.plot_moving_average_over_width(data, 9, project, data_type)
            self.plot_moving_average_over_width(data, 19, project, data_type)

    def shear_rate_analysis(self, end_iteration: int, n_iterations_before: int = 1):
        for project in self.projects:
            self.set_x_ticks(self.x_ticks_hdf5[project])

            self.status.print(f"Loading shear rate data of {project}")
            iterations = list(filter(lambda i: i <= end_iteration, self.hdf5_iterations[project]))[-n_iterations_before:]

            bulk_load = BulkCollector(
                self.connection,
                self.data[project].id,
                {i: self.data[project].hdf5_iterations[i] for i in iterations},
                {},
                self.status
            )

            shear_rate_data = [row.exclude_borders(self.data[project].boundary) for row in bulk_load.fluid_shear_rate()]
            self.status.println(f"Loaded shear rate data of {project}")

            data = nanmean([data.magnitude for data in shear_rate_data], axis=0)
            data_type = 'shear rate'

            self.plot_vertical_slice(data, project, data_type)
            self.plot_horizontal_slice(data, project, data_type)
            self.plot_moving_average_over_height(data, 5, project, data_type)
            self.plot_moving_average_over_height(data, 9, project, data_type)
            self.plot_moving_average_over_height(data, 19, project, data_type)
            self.plot_moving_average_over_width(data, 5, project, data_type)
            self.plot_moving_average_over_width(data, 9, project, data_type)
            self.plot_moving_average_over_width(data, 19, project, data_type)