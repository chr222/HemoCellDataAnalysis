import numpy as np
import params
from pathlib import Path
import math

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

    def fluid_velocity_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5):
        self.set_x_ticks(self.x_ticks_hdf5)
        self.status.print("Loading velocity data")

        slice_in = round(len(self.data.boundary.boundary_map)*xslice)

        velocity_data = list(map(lambda x: x.exclude_borders(self.data.boundary), self.data.all.fluid_velocities()[start_it:end_it]))

        for i in range(len(velocity_data)):
            velocity_data[i][:] = velocity_data[i][slice_in]

        self.status.println("Loaded velocity data")

        velocity = np.array([data.magnitude for data in velocity_data]) * 1e3
        velocity = np.mean(velocity, axis=0)
        velocity = np.mean(velocity, axis=0)  # remove x-coordinates (same bc slice) TODO: replace

        self.plot_time_averaged_cross_section(velocity, 'fluid', 'velocity')  # TODO: include averaged from until

    def fluid_shearrate_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5):
        self.set_x_ticks(self.x_ticks_hdf5)
        self.status.print("Loading shear-rate data")
        slice_in = round(len(self.data.boundary.boundary_map) * xslice)
        sr_data = list(map(lambda x: x.exclude_borders(self.data.boundary), self.data.all.fluid_shear_rate()[start_it:end_it]))

        for i in range(len(sr_data)):
            sr_data[i][:] = sr_data[i][slice_in]

        self.status.println("Loaded shear-rate data")

        shear_rate = np.array([np.float32(data).magnitude for data in sr_data])
        shear_rate = np.mean(shear_rate, axis=0)
        shear_rate = np.mean(shear_rate, axis=0)

        self.plot_time_averaged_cross_section(shear_rate, 'fluid', 'shear rate')

    def fluid_elongrate_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5, peak: bool = False):
        self.set_x_ticks(self.x_ticks_hdf5)
        self.status.print("Loading elongation-rate data")
        slice_in = round(len(self.data.boundary.boundary_map) * xslice)

        elong_data = list(
            map(lambda x: x.exclude_borders(self.data.boundary), self.data.all.fluid_strain_rate()[start_it:end_it]))

        for i in range(len(elong_data)):
            elong_data[i][:] = elong_data[i][slice_in]

        self.status.println("Loaded elongation-rate data")

        elong_rate = np.array([np.float32(data).x_elong for data in elong_data])
        elong_rate = np.mean(elong_rate, axis=0)  # average timesteps
        elong_rate = np.mean(elong_rate, axis=0)  # remove x-coordinates (same bc slice) TODO replace

        if peak:
            print(np.amax(elong_rate, axis=1))
            print(np.amax(elong_rate, axis=0))
            print(np.amax(elong_rate, axis=2))

        self.plot_time_averaged_cross_section(elong_rate, 'fluid', 'elongation rate')

    def cell_ratio_comparison(self, iteration: int = -1, x_start: float = 0, x_end: float = 1, y_start: float = 0, y_end: float = 1, z_start: float = 0, z_end: float = 1):
        self.set_x_ticks(self.x_ticks_csv)

        x_max = 300e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        y_max = 150e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        z_max = 178e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        x_from = x_start * x_max
        x_to = x_end * x_max
        y_from = y_start * y_max
        y_to = y_end * y_max
        z_from = z_start * z_max
        z_to = z_end * z_max
        self.status.print("Loading RBC positions")
        rbc_positions = [self.data.all.red_blood_cell_positions()[iteration]]
        self.status.println("Loaded RBC positions")
        rbc_check = 0
        for i in (rbc_positions[0]):
            if rbc_positions[0][i][2] >= z_from:
                rbc_check = rbc_check+1
        print(rbc_check)

        self.status.print("Loading PLT positions")
        plt_positions = [self.data.all.platelet_positions()[iteration]]
        self.status.println("Loaded PLT positions")
        plt_check = 0
        for i in (plt_positions[0]):
            if plt_positions[0][i][2] >= z_from:
                plt_check = plt_check + 1
        print(plt_check)


    def local_hematocrit_calc(self, iteration: int = -1, x_start: float = 0, x_end: float = 1, y_start: float = 0,
                              y_end: float = 1, z_start: float = 0, z_end: float = 1):
        self.set_x_ticks(self.x_ticks_csv)
        rbc_volume = 90 # cubic micrometers
        x_max = 300e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        y_max = 150e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        z_max = 178e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        x_from = x_start * x_max
        x_to = x_end * x_max
        y_from = y_start * y_max
        y_to = y_end * y_max
        z_from = z_start * z_max
        z_to = z_end * z_max
        bb_volume = ((x_to-x_from)*1e06)*((y_to-y_from)*1e06)*((z_to-z_from)*1e06)
        print(bb_volume)
        self.status.print("Loading RBC positions")
        rbc_positions = [self.data.all.red_blood_cell_positions()[iteration]]
        self.status.println("Loaded RBC positions")
        rbc_count = 0
        for i in (rbc_positions[0]):
            if rbc_positions[0][i][0] >= x_from and rbc_positions[0][i][0] < x_to and rbc_positions[0][i][1] >= y_from and rbc_positions[0][i][1] < y_to and rbc_positions[0][i][2] >= z_from and rbc_positions[0][i][2] < z_to:
                rbc_count = rbc_count + 1
        print(rbc_count)
        hematocrit = (rbc_count*rbc_volume)/bb_volume
        print(hematocrit)