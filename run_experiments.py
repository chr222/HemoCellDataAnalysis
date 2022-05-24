import multiprocessing as mp
from src.experiments import Experiments


def main():
    # Load simulation into the Experiment class
    experiments = Experiments(
        project_name='puncture-db-50',
        save_figures=True
    )

    # Run experiments via processes (this helps to clean up the data after the experiment is finished)
    for analysis in [
        #experiments.fluid_velocity_analysis,
        #experiments.cell_velocity_analysis,
        #experiments.fluid_velocity_time_averaged_crossx(start_it=15, end_it=27),
        #experiments.fluid_shearrate_time_averaged_crossx(start_it=15, end_it=27),
        experiments.fluid_elongrate_time_averaged_crossx(start_it=15, end_it=27, peak=True)
        #experiments.cell_ratio_comparison(iteration=27, z_start=0.562)
        #experiments.local_hematocrit_calc(iteration =27, x_start=3/8, x_end=5/8, y_start=1/3, y_end=2/3, z_start=103/178,
        #                                  z_end=168/178)
        # 112.5 - 187.5 = 75; 50 - 100 = 50; 103 - 168 = 65
        #experiments.local_hematocrit_calc(x_start=0.417, x_end=0.583, y_start=0.333, y_end=0.667, z_start=0.579, z_end=0.944)
        #125 - 175 = 50; 50 - 100 = 50; 103 - 168 = 65
    ]:
        analysis()
        #p = mp.Process(target=analysis)
        #p.start()
        #p.join()


if __name__ == '__main__':
    main()
