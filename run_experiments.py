import multiprocessing as mp
from experiments import Experiments


def main():
    # Load simulation into the Experiment class
    experiments = Experiments(
        project_name='template_project',
        save_figures=True
    )

    # Run experiments via processes (this helps to clean up the data after the experiment is finished)
    for analysis in [
        experiments.fluid_velocity_analysis,
        experiments.cell_velocity_analysis
    ]:
        p = mp.Process(target=analysis)
        p.start()
        p.join()


if __name__ == '__main__':
    main()
