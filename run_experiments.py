import multiprocessing as mp
from src.experiments import Experiments


def main():
    # Load simulation into the Experiment class
    experiments = Experiments(
        project_name='wallstent-db', #puncture-db-75 puncture-db-250Pa-75
        save_figures=True
    )
    x_limit = (0, 250)  # Set your desired x-axis limits in um
    y_limit = (0, 137.5)  # Set your desired y-axis limits
    name = 'wallstent-viridis'
    # Run experiments via processes (this helps to clean up the data after the experiment is finished)
    for analysis in [
        #experiments.fluid_velocity_time_averaged_crossx(start_it=-2, end_it=-1) #, scale_max=6
        #experiments.fluid_shearrate_time_averaged_crossx(start_it=-2, end_it=-1, peak=True), #, scale_max=500
        #experiments.fluid_elongrate_time_averaged_crossx(start_it=-2, end_it=-1, peak=True)  #, scale_max=150
        #experiments.cell_ratio_comparison_time_averaged(start_it=-36, end_it=-1, x_end=50, z_end=100, section_name="inlet"), #inlet x_end = 50 (50um) 37.5 (75um)
        #experiments.cell_ratio_comparison_time_averaged(start_it=-36, end_it=-1, z_start=100, section_name="outlet chamber"), #outlet chamber
        #experiments.cell_ratio_comparison_time_averaged(start_it=-42, end_it=-7, x_start=100, z_end=100, section_name="outlet downstream"), #outlet downstream x_start = 100 (50um) 112.5 (75um)
        #experiments.puncture_cell_conc_time_averaged(start_it=-42, end_it=-7, pun_diam=50, time_window=5),
        #experiments.local_cfl_calc_time_averaged(start_it=-42, end_it=-7, pun_diam=50, time_window=5)
        #experiments.fluid_velocity_time_averaged_side(start_it=0, end_it=-1, lab=name, scale_max=39.53),  # , scale_max=500, x_limit, y_limit
        #experiments.fluid_stream_time_averaged_side(start_it=0, end_it=-1, lab=name),
        experiments.fluid_shearrate_time_averaged_side(start_it=0, end_it=-1, lab=name, scale_max=1631),  # , scale_max=500, x_limit, y_limit
        experiments.fluid_elongrate_time_averaged_side(start_it=0, end_it=-1, lab=name, scale_max=344)  # , scale_max=500, x_limit, y_limit
    ]:
        analysis()
        #p = mp.Process(target=analysis)
        #p.start()
        #p.join()


if __name__ == '__main__':
    main()
