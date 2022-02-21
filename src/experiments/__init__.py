import numpy as np
import params
from pathlib import Path

from src.graphics import Graphics
from src.progress import StatusHandler
from src.sql.connection import Connection
from src.sql.entity.simulation import load_simulation


class Experiments(Graphics):
    def __init__(self, project_name: str, save_figures: bool):
        self.status = StatusHandler()

        self.status.print('Connecting to database...')
        self.connection = Connection(Path(params.DATABASE_NAME), Path(params.MATRIX_DIRECTORY))
        self.status.println('Connection successful')

        self.status.print('Loading simulation...')
        self.data = load_simulation(self.connection, project_name, self.status)
        self.status.println('Simulation loaded')

        self.status.print('Loading simulation meta data...')
        self.hdf5_iterations = sorted(list(self.data.hdf5_iterations.keys()))
        self.x_ticks_hdf5 = self.get_x_ticks(self.hdf5_iterations)
        self.csv_iterations = sorted(list(self.data.csv_iterations.keys()))
        self.x_ticks_csv = self.get_x_ticks(self.csv_iterations)
        self.status.println('Meta data loaded')

        # Init Graphics
        super().__init__(Path(params.EXPERIMENT_OUTPUT_DIRECTORY), save_figures)

    def get_x_ticks(self, x) -> np.ndarray:
        return np.around(np.array(x) * self.data.config.dt / 1e-3, decimals=1)

    def fluid_velocity_analysis(self):
        self.set_x_ticks(self.x_ticks_hdf5)

        self.status.print("Loading velocity data")
        velocity_data = list(map(lambda x: x.exclude_borders(self.data.boundary), self.data.all.fluid_velocities()))
        self.status.println("Loaded velocity data")

        velocity = np.array([data.magnitude for data in velocity_data]) * 1e6

        self.plot_mean_over_time(velocity, 'fluid', 'velocity')

    def cell_velocity_analysis(self):
        self.set_x_ticks(self.x_ticks_csv)

        self.status.print("Loading RBC velocities")
        rbc_velocities = [row.scale(1e6) for row in self.data.all.red_blood_cell_velocities()]
        self.status.println("Loaded RBC velocities")

        self.status.print("Loading PLT velocities")
        plt_velocities = [row.scale(1e6) for row in self.data.all.platelet_velocities()]
        self.status.println("Loaded PLT velocities")

        rbc_velocity = [list(row.magnitude.values()) for row in rbc_velocities]
        plt_velocity = [list(row.magnitude.values()) for row in plt_velocities]

        self.csv_plot_mean_over_time(rbc_velocity, label='RBC', color='r')
        self.csv_plot_mean_over_time(plt_velocity, prefix='cell', data_type='velocity', label='PLT', color='b')

