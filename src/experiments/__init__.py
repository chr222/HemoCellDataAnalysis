import numpy as np
import params
import warnings

from src.graphics import Graphics
from src.progress import StatusHandler
from src.sql.connection import Connection
from src.sql.entity.simulation import load_simulation


class Experiments(Graphics):
    def __init__(self, project_name: str, save_figures: bool):
        self.status = StatusHandler()

        self.status.print('Connecting to database...')
        self.connection = Connection(params.DATABASE_NAME)
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
        super().__init__(project_name, save_figures)

    @staticmethod
    def np_nan_mean(a: np.ndarray, axis=None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(a, axis)

    def get_x_ticks(self, x) -> np.ndarray:
        return np.around(np.array(x) * self.data.config.dt / 1e-3, decimals=1)

    def fluid_velocity_analysis(self):
        self.set_x_ticks(self.x_ticks_hdf5)

        self.status.print("Loading velocity data")
        velocity_data = list(map(lambda x: x.exclude_borders(self.data.boundary), self.data.all.fluid_velocities()))
        self.status.println("Loaded velocity data")

        velocity = np.array([data.magnitude for data in velocity_data]) * 1e6
        velocity_y = np.absolute([data.y for data in velocity_data]) * 1e6

        for (data, data_type) in [
            (velocity, 'velocity'),
            (velocity_y, 'velocity-y')
        ]:
            self.plot_mean_over_time(data, 'fluid', data_type)
            self.plot_median_over_time(data, 'fluid', data_type)
            self.plot_max_over_time(data, 'fluid', data_type)
            self.plot_mean_std_over_time(data, 'fluid', data_type)
            self.plot_mean_minmax_over_time(data, 'fluid', data_type)
            self.plot_median_std_over_time(data, 'fluid', data_type)
            self.plot_median_minmax_over_time(data, 'fluid', data_type)
            self.plot_mean_over_height_over_time(data, 'fluid', data_type)
            self.plot_median_over_height_over_time(data, 'fluid', data_type)
            self.plot_max_over_height_over_time(data, 'fluid', data_type)
            self.plot_mean_over_radius_over_time(data, 'fluid', data_type)
            self.plot_median_over_radius_over_time(data, 'fluid', data_type)
            self.plot_max_over_radius_over_time(data, 'fluid', data_type)
            self.plot_moving_average_over_time(data, 5, 'fluid', data_type)
            self.plot_moving_average_over_time(data, 11, 'fluid', data_type)

    def fluid_shear_stress_analysis(self):
        self.set_x_ticks(self.x_ticks_hdf5)

        self.status.print("Loading shear stress data")
        data = np.array([data.exclude_borders(self.data.boundary).magnitude for data in self.data.all.fluid_shear_stress()])
        self.status.println("Loaded shear stress data")

        data_type = 'shear stress'

        self.plot_mean_over_time(data, 'fluid', data_type)
        self.plot_median_over_time(data, 'fluid', data_type)
        self.plot_max_over_time(data, 'fluid', data_type)
        self.plot_mean_std_over_time(data, 'fluid', data_type)
        self.plot_mean_minmax_over_time(data, 'fluid', data_type)
        self.plot_median_std_over_time(data, 'fluid', data_type)
        self.plot_median_minmax_over_time(data, 'fluid', data_type)
        self.plot_mean_over_height_over_time(data, 'fluid', data_type)
        self.plot_median_over_height_over_time(data, 'fluid', data_type)
        self.plot_max_over_height_over_time(data, 'fluid', data_type)
        self.plot_mean_over_radius_over_time(data, 'fluid', data_type)
        self.plot_median_over_radius_over_time(data, 'fluid', data_type)
        self.plot_max_over_radius_over_time(data, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 5, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 11, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 21, 'fluid', data_type)

    def fluid_shear_rate_analysis(self):
        self.set_x_ticks(self.x_ticks_hdf5)

        self.status.print("Loading shear rate data")
        data = np.array([data.exclude_borders(self.data.boundary).magnitude for data in self.data.all.fluid_shear_rate()])
        self.status.println("Loaded shear rate data")

        data_type = 'shear rate'

        self.plot_mean_over_time(data, 'fluid', data_type)
        self.plot_median_over_time(data, 'fluid', data_type)
        self.plot_max_over_time(data, 'fluid', data_type)
        self.plot_mean_std_over_time(data, 'fluid', data_type)
        self.plot_mean_minmax_over_time(data, 'fluid', data_type)
        self.plot_median_std_over_time(data, 'fluid', data_type)
        self.plot_median_minmax_over_time(data, 'fluid', data_type)
        self.plot_mean_over_height_over_time(data, 'fluid', data_type)
        self.plot_median_over_height_over_time(data, 'fluid', data_type)
        self.plot_max_over_height_over_time(data, 'fluid', data_type)
        self.plot_mean_over_radius_over_time(data, 'fluid', data_type)
        self.plot_median_over_radius_over_time(data, 'fluid', data_type)
        self.plot_max_over_radius_over_time(data, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 5, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 11, 'fluid', data_type)
        self.plot_moving_average_over_time(data, 21, 'fluid', data_type)

    def hematocrit_analysis(self):
        self.set_x_ticks(self.x_ticks_hdf5)

        self.status.print("Loading hematocrits")
        hematocrits = np.array(self.data.all.hematocrits()) * 100.0
        self.status.println("Loaded hematocrits")

        self.plot_mean_over_height_over_time(hematocrits, 'fluid', 'hematocrit')
        self.plot_mean_over_radius_over_time(hematocrits, 'fluid', 'hematocrit')
        self.plot_median_over_height_over_time(hematocrits, 'fluid', 'hematocrit')
        self.plot_median_over_radius_over_time(hematocrits, 'fluid', 'hematocrit')

    def cell_count_analysis(self):
        self.set_x_ticks(self.x_ticks_csv)

        self.status.print("Loading RBC counts")
        rbc_counts = np.array(self.data.all.red_blood_cell_counts())
        self.status.println("Loaded RBC counts")

        self.status.print("Loading PLT counts")
        plt_counts = np.array(self.data.all.platelet_counts())
        self.status.println("Loaded PLT counts")

        self.plot_over_time(rbc_counts, label='RBC', color='r')
        self.plot_over_time(plt_counts, 'cell', 'count', label='PLT', color='b')

    def cell_velocity_analysis(self):
        self.set_x_ticks(self.x_ticks_csv)

        self.status.print("Loading RBC velocities")
        rbc_velocities = [row.scale(1e6) for row in self.data.all.red_blood_cell_velocities()]
        self.status.println("Loaded RBC velocities")

        self.status.print("Loading PLT velocities")
        plt_velocities = [row.scale(1e6) for row in self.data.all.platelet_velocities()]
        self.status.println("Loaded PLT velocities")

        rbc_velocity = [list(row.magnitude.values()) for row in rbc_velocities]
        rbc_velocity_y = [list(map(abs, row.y.values())) for row in rbc_velocities]
        plt_velocity = [list(row.magnitude.values()) for row in plt_velocities]
        plt_velocity_y = [list(map(abs, row.y.values())) for row in plt_velocities]

        for rbc_data, plt_data, data_type in [
            (rbc_velocity, plt_velocity, 'velocity'),
            (rbc_velocity_y, plt_velocity_y, 'velocity-y')
        ]:
            self.csv_plot_mean_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_max_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_max_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_mean_std_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_std_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_mean_minmax_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_minmax_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_std_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_std_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_minmax_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_minmax_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 5, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 5, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 11, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 11, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 21, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 21, 'cell', data_type, label='PLT', color='b')

    def cell_position_analysis(self):
        self.set_x_ticks(self.x_ticks_csv)

        self.status.print("Loading RBC positions")
        rbc_positions = [row.scale(1e6) for row in self.data.all.red_blood_cell_positions()]
        self.status.println("Loaded RBC positions")

        self.status.print("Loading PLT positions")
        plt_positions = [row.scale(1e6) for row in self.data.all.platelet_positions()]
        self.status.println("Loaded PLT positions")

        width = self.data.config.nx / 2

        rbc_height = [list(row.y.values()) for row in rbc_positions]
        rbc_distance_from_center = [list(row.distance_from_center(width).values()) for row in rbc_positions]
        rbc_start_y = rbc_positions[0].y
        rbc_vertical_movement = [[y - rbc_start_y[cell_id] for cell_id, y in row.y.items()] for row in rbc_positions]
        rbc_start_d = rbc_positions[0].distance_from_center(width)
        rbc_horizontal_movement = [[d - rbc_start_d[cell_id] for cell_id, d in row.distance_from_center(width).items()] for row in rbc_positions]

        plt_height = [list(row.y.values()) for row in plt_positions]
        plt_distance_from_center = [list(row.distance_from_center(width).values()) for row in plt_positions]
        plt_start_y = plt_positions[0].y
        plt_vertical_movement = [[y - plt_start_y[cell_id] for cell_id, y in row.y.items()] for row in plt_positions]
        plt_start_d = plt_positions[0].distance_from_center(width)
        plt_horizontal_movement = [[d - plt_start_d[cell_id] for cell_id, d in row.distance_from_center(width).items()] for row in plt_positions]

        for rbc_data, plt_data, data_type in [
            (rbc_height, plt_height, 'height'),
            (rbc_distance_from_center, plt_distance_from_center, 'distance from center'),
            (rbc_vertical_movement, plt_vertical_movement, 'vertical movement'),
            (rbc_horizontal_movement, plt_horizontal_movement, 'horizontal movement')
        ]:
            self.csv_plot_mean_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_max_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_max_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_mean_std_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_std_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_mean_minmax_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_mean_minmax_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_std_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_std_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_median_minmax_over_time(rbc_data, label='RBC', color='r')
            self.csv_plot_median_minmax_over_time(plt_data, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 5, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 5, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 11, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 11, 'cell', data_type, label='PLT', color='b')

            self.csv_plot_moving_average_over_time(rbc_data, 21, label='RBC', color='r')
            self.csv_plot_moving_average_over_time(plt_data, 21, 'cell', data_type, label='PLT', color='b')
