import multiprocessing as mp

from src.experiments.comparison_experiments import ComparisonExperiments
from src.experiments import Experiments

projects_cone_angle = [
    'impactr_20_0_8',
    'impactr_20_2_8',
    'impactr_20_4_8',
    'impactr_20_8_8',
    'impactr_20_16_8'
]

projects_frequency = [
    'impactr_20_8_2',
    'impactr_20_8_4',
    'impactr_20_8_8',
    'impactr_20_8_16',
]


def main():
    # for project in set(projects_cone_angle + projects_frequency):
    #     print(f"\nRunning experiments on project {project}")
    #     experiments = Experiments(
    #         project_name=project,
    #         save_figures=True
    #     )
    #
    #     for analysis in [
    #         experiments.fluid_velocity_analysis,
    #         experiments.fluid_shear_stress_analysis,
    #         experiments.fluid_shear_rate_analysis,
    #         experiments.hematocrit_analysis,
    #         experiments.cell_velocity_analysis,
    #         experiments.cell_position_analysis,
    #         experiments.cell_count_analysis
    #     ]:
    #         p = mp.Process(target=analysis)
    #         p.start()
    #         p.join()
    #
    # for output_name, projects in [
    #     ("cone_angle", projects_cone_angle),
    #     ("frequency", projects_frequency[:-1])
    # ]:
    #     print(f"\nRunning experiments on project {output_name}")
    #     experiments = ComparisonExperiments(output_name, projects, save_figures=True)
    #     iteration = 2975000  # Max iteration that all simulation reached
    #
    #     for analysis in [
    #         experiments.velocity_analysis,
    #         experiments.shear_stress_analysis,
    #         experiments.shear_rate_analysis
    #     ]:
    #         p = mp.Process(target=analysis, args=(iteration, 40))
    #         p.start()
    #         p.join()

    for output_name, projects in [
        ("cone_angle_single", projects_cone_angle),
        ("frequency_single", projects_frequency[:-1])
    ]:
        print(f"\nRunning experiments on project {output_name}")
        experiments = ComparisonExperiments(output_name, projects, save_figures=True)
        iteration = 2500000  # Max iteration that all simulation reached

        for analysis in [
            experiments.velocity_analysis,
            experiments.shear_stress_analysis,
            experiments.shear_rate_analysis
        ]:
            p = mp.Process(target=analysis, args=(iteration, 1))
            p.start()
            p.join()


if __name__ == '__main__':
    main()
