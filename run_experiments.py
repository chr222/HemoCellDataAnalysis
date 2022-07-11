import multiprocessing as mp
from src.experiments import Experiments


def main():
    # Load simulation into the Experiment class
    experiments = Experiments(
        project_name='puncture-db-250Pa-75', #puncture-db-75 puncture-db-250Pa-75
        save_figures=True
    )

    # Run experiments via processes (this helps to clean up the data after the experiment is finished)
    for analysis in [
        #experiments.fluid_velocity_time_averaged_crossx(start_it=0, end_it=-1, scale_max=3),
        #experiments.fluid_shearrate_time_averaged_crossx(start_it=0, end_it=-1, scale_max=900),
        #experiments.fluid_elongrate_time_averaged_crossx(start_it=0, end_it=-1, peak=True, scale_max=350),
        experiments.cell_ratio_comparison_time_averaged(start_it=0, end_it=-1, x_end=37.5, z_end=100, section_name="inlet"), #inlet x_end = 50 (50um) 37.5 (75um)
        experiments.cell_ratio_comparison_time_averaged(start_it=0, end_it=-1, z_start=100, section_name="outlet chamber"), #outlet chamber
        experiments.cell_ratio_comparison_time_averaged(start_it=0, end_it=-1, x_start=112.5, z_end=100, section_name="outlet downstream"), #outlet downstream x_start = 100 (50um) 112.5 (75um)
        experiments.puncture_cell_conc_time_averaged(start_it=0, end_it=-1, pun_diam=75, time_window=5),
        experiments.local_cfl_calc_time_averaged(start_it=0, end_it=-1, pun_diam=75, time_window=5)
    ]:
        analysis()
        #p = mp.Process(target=analysis)
        #p.start()
        #p.join()


if __name__ == '__main__':
    main()
