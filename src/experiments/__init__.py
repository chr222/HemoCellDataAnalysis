import numpy as np
import params
from pathlib import Path
import math
import statistics

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

    def fluid_velocity_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5, scale_max: int or None=None):
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

        self.plot_time_averaged_cross_section(velocity, 'fluid', 'velocity', v_max=scale_max)  # TODO: include averaged from until

    def fluid_shearrate_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5, scale_max: int or None=None):
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

        self.plot_time_averaged_cross_section(shear_rate, 'fluid', 'shear rate', v_max=scale_max)

    def fluid_elongrate_time_averaged_crossx(self, start_it: int = 0, end_it: int = -1, xslice: float = 0.5, peak: bool = False, scale_max: int or None=None):
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

        #if peak:
        #    print(np.amax(elong_rate, axis=1))
        #    print(np.amax(elong_rate, axis=0))
        #    print(np.amax(elong_rate, axis=2))

        self.plot_time_averaged_cross_section(elong_rate, 'fluid', 'elongation rate', v_max=scale_max)

    def cell_ratio_comparison(self, iteration: int = -1, x_start: float = 0, x_end: float = 1, y_start: float = 0, y_end: float = 1, z_start: float = 0, z_end: float = 1):
        self.set_x_ticks(self.x_ticks_csv)

        x_max = 300e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        y_max = 100e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        z_max = 157.5e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
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
            if rbc_positions[0][i][0] >= x_from and rbc_positions[0][i][0] <= x_to and rbc_positions[0][i][1] >= y_from and rbc_positions[0][i][1] <= y_to and rbc_positions[0][i][2] >= z_from and rbc_positions[0][i][2] <= z_to:
                rbc_check = rbc_check+1
        print(rbc_check)

        self.status.print("Loading PLT positions")
        plt_positions = [self.data.all.platelet_positions()[iteration]]
        self.status.println("Loaded PLT positions")
        plt_check = 0
        for i in (plt_positions[0]):
            if plt_positions[0][i][0] >= x_from and plt_positions[0][i][0] <= x_to and plt_positions[0][i][1] >= y_from and plt_positions[0][i][1] <= y_to and plt_positions[0][i][2] >= z_from and plt_positions[0][i][2] <= z_to:
                plt_check = plt_check + 1
        print(plt_check)


    def cell_ratio_comparison_time_averaged(self, start_it: int = 0, end_it: int = -1, x_start: float = 0, x_end: float = 300, y_start: float = 0, y_end: float = 100, z_start: float = 0, z_end: float = 157.5, section_name: str="undefined"):
        self.set_x_ticks(self.x_ticks_csv)

        #x_max = 300e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        #y_max = 100e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        #z_max = 157.5e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        lattice_offset = 0.75e-06
        x_from = x_start*(10**(-6)) - lattice_offset
        x_to = x_end*(10**(-6)) - lattice_offset
        y_from = y_start*(10**(-6)) - lattice_offset
        y_to = y_end*(10**(-6)) - lattice_offset
        z_from = z_start*(10**(-6)) - lattice_offset
        z_to = z_end*(10**(-6)) - lattice_offset
        transform_index = range(0,len(self.data.all.red_blood_cell_positions()))  # in case negative indeces are used for start_it or end_it
        rbc_counter, plt_counter = [], []
        for iteration in range(transform_index[start_it], transform_index[end_it]+1):
            self.status.print("Iteration: "+str(iteration))
            rbc_positions = [self.data.all.red_blood_cell_positions()[iteration]]
            rbc_check = 0
            for i in (rbc_positions[0]):
                if rbc_positions[0][i][0] >= x_from and rbc_positions[0][i][0] <= x_to and rbc_positions[0][i][1] >= y_from and rbc_positions[0][i][1] <= y_to and rbc_positions[0][i][2] >= z_from and rbc_positions[0][i][2] <= z_to:
                    rbc_check = rbc_check+1
            rbc_counter.append(rbc_check)

            plt_positions = [self.data.all.platelet_positions()[iteration]]
            plt_check = 0
            for i in (plt_positions[0]):
                if plt_positions[0][i][0] >= x_from and plt_positions[0][i][0] <= x_to and plt_positions[0][i][1] >= y_from and plt_positions[0][i][1] <= y_to and plt_positions[0][i][2] >= z_from and plt_positions[0][i][2] <= z_to:
                    plt_check = plt_check + 1
            plt_counter.append(plt_check)
        rbc_counter = np.array(rbc_counter)
        plt_counter = np.array(plt_counter)
        ratio = np.divide(rbc_counter, plt_counter)
        print("RBC:platelet ration at "+section_name+" = "+str(np.mean(ratio))+" +- "+str(np.std(ratio)))


    def puncture_cell_conc_time_averaged(self, start_it: int = 0, end_it: int = -1, z_from: float = 100, z_to: float = 107.5, x_center: float = 75, pun_diam: float = 50, time_window: float = 1):
        #pun_diam in micrometers  old:z_start: float = 0.635, z_end: float = 0.683, pun_origin: float = 0.417
        self.set_x_ticks(self.x_ticks_csv)
        rbc_volume = 90  # cubic micrometers
        plt_volume = 11  # cubic micrometers
        lattice_offset = 0.75e-06
        wound_neck_volume = (pun_diam*0.5)**2*((z_to-z_from))*math.pi  # in micrometers cubed
        transform_index = range(0,
                                len(self.data.all.red_blood_cell_positions()))  # in case negative indeces are used for start_it or end_it
        rbc_conc_prox, plt_conc_prox, rbc_conc_dist, plt_conc_dist = [], [], [], []
        rbc_check_prox_list, plt_check_prox_list, rbc_check_dist_list, plt_check_dist_list = [], [], [], []
        for iteration in range(transform_index[start_it], transform_index[end_it] + 1):
            self.status.print("Iteration: " + str(iteration))
            rbc_positions = [self.data.all.red_blood_cell_positions()[iteration]]
            rbc_check_prox, rbc_check_dist = 0, 0
            for i in (rbc_positions[0]):
                if rbc_positions[0][i][2] > (z_from*(10**(-6)) - lattice_offset) and rbc_positions[0][i][2] <= (z_to*(10**(-6)) - lattice_offset):
                    if rbc_positions[0][i][0] < x_center*(10**(-6))-lattice_offset:
                        rbc_check_prox = rbc_check_prox + 1
                    elif rbc_positions[0][i][0] > x_center*(10**(-6))-lattice_offset:
                        rbc_check_dist = rbc_check_dist + 1
            rbc_check_prox_list.append(rbc_check_prox)
            rbc_check_dist_list.append(rbc_check_dist)
            if iteration % time_window == 0 or iteration == transform_index[end_it]:
                rbc_conc_prox.append((np.mean(np.array(rbc_check_prox_list))*rbc_volume)/(wound_neck_volume/2))
                rbc_conc_dist.append((np.mean(np.array(rbc_check_dist_list))* rbc_volume) / (wound_neck_volume / 2))

            plt_positions = [self.data.all.platelet_positions()[iteration]]
            plt_check_prox, plt_check_dist = 0, 0
            for i in (plt_positions[0]):
                if plt_positions[0][i][2] > z_from*(10**(-6)) and plt_positions[0][i][2] <= z_to*(10**(-6)):
                    if plt_positions[0][i][0] < x_center*(10**(-6)):
                        plt_check_prox = plt_check_prox + 1
                    elif plt_positions[0][i][0] > x_center*(10**(-6)):
                        plt_check_dist = plt_check_dist + 1
            plt_check_prox_list.append(plt_check_prox)
            plt_check_dist_list.append(plt_check_dist)
            if iteration % time_window == 0 or iteration == transform_index[end_it]:
                plt_conc_prox.append((np.mean(np.array(plt_check_prox_list)) * plt_volume) / (wound_neck_volume / 2))
                plt_conc_dist.append((np.mean(np.array(plt_check_dist_list)) * plt_volume) / (wound_neck_volume / 2))

        print("RBC volume concentration proximal side: " + str(np.mean(np.array(rbc_conc_prox))) + " +- " + str(
            np.std(np.array(rbc_conc_prox))))
        print("RBC volume concentration distal side: " + str(np.mean(np.array(rbc_conc_dist))) + " +- " + str(
            np.std(np.array(rbc_conc_dist))))
        print("PLT volume concentration proximal side: " + str(np.mean(np.array(plt_conc_prox))) + " +- " + str(
            np.std(np.array(plt_conc_prox))))
        print("PLT volume concentration distal side: " + str(np.mean(np.array(plt_conc_dist))) + " +- " + str(
            np.std(np.array(plt_conc_dist))))


    def local_cfl_calc_time_averaged(self, start_it: int = 0, end_it: int = -1, z_from: float = 100, z_to: float = 107.5, x_center: float = 75, pun_diam=50, time_window: float = 5, threshold = 0.05):
        self.set_x_ticks(self.x_ticks_csv)

        #x_max = 300e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        y_max = 100e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        #z_max = 157.5e-06  # TODO: get domain dimensions and overwrite 'start'/'stop' coordinates
        lattice_offset = 0.75e-06
        pun_origin_xy = np.array([(x_center*(10**(-6)))-lattice_offset, (y_max / 2)-lattice_offset])
        transform_index = range(0,
                                len(self.data.all.red_blood_cell_positions()))  # in case negative indeces are used for start_it or end_it
        cfl_prox_quarter, cfl_dist_quarter = [], []
        rbc_check = 0
        cells_prox_quarter, cells_dist_quarter = [], []
        cell_id_list = []
        iteration_counter = 0
        for iteration in range(transform_index[start_it], transform_index[end_it] + 1):
            iteration_counter = iteration_counter+1
            self.status.print("Iteration: " + str(iteration))
            rbc_positions = [self.data.all.red_blood_cell_positions()[iteration]]
            for i in (rbc_positions[0]):
                if z_from*(10**(-6))-lattice_offset < rbc_positions[0][i][2] <= z_to*(10**(-6))-lattice_offset:
                    if i in cell_id_list:
                        continue
                    cell_id_list.append(i)
                    rbc_check = rbc_check + 1
                    tmp_xy = np.array([rbc_positions[0][i][0]-pun_origin_xy[0], rbc_positions[0][i][1]-pun_origin_xy[1]])
                    tmp_r = math.sqrt(tmp_xy[0]**2+tmp_xy[1]**2)
                    if math.cos(math.radians(135))*tmp_r < tmp_xy[0] <= math.cos(math.radians(45))*tmp_r and math.sin(math.radians(135))*tmp_r < tmp_xy[1]:
                        # distal quarter
                        cells_dist_quarter.append(math.sin(math.radians(90))*tmp_r)
                    elif math.cos(math.radians(225))*tmp_r < tmp_xy[0] <= math.cos(math.radians(315))*tmp_r and math.sin(math.radians(225))*tmp_r > tmp_xy[1]:
                        # proximal quarter
                        cells_prox_quarter.append(math.sin(math.radians(270))*tmp_r)
            #print("rbc_check: " + str(rbc_check))
            if iteration_counter % time_window == 0 or iteration == transform_index[end_it]:
                threshold_cells = round(threshold * rbc_check / 4)  # absolute threshold in amount of cells outside of CFL per quarter
                #print("threshold_cells: " + str(threshold_cells))
                if threshold_cells == 0:
                    threshold_cells = 1
                cfl_prox_quarter.append(pun_diam*(10**(-6))/2-(abs(sorted(cells_prox_quarter)[threshold_cells-1])))
                cfl_dist_quarter.append(pun_diam*(10**(-6))/2-(sorted(cells_dist_quarter)[-threshold_cells]))
                rbc_check = 0
                cells_prox_quarter, cells_dist_quarter = [], []
                cell_id_list = []

        print("CFL proximal quarter: " + str(np.mean(np.array(cfl_prox_quarter))) + " +- " + str(
            np.std(np.array(cfl_prox_quarter))))
        print("CFL distal quarter: " + str(np.mean(np.array(cfl_dist_quarter))) + " +- " + str(
            np.std(np.array(cfl_dist_quarter))))

