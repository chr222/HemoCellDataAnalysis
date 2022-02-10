from src.sql.entity.boundary import Boundary, load_boundary
from src.sql.entity.bulk_collector import BulkCollector
from src.sql.entity.config import create_config, load_config, Config
from src.sql.entity.csv_cell import create_csv_cells
from src.sql.entity.csv_iteration import CSVIteration, load_csv_iterations
from src.sql.entity.hdf5_cell import create_hdf5_cells
from src.sql.entity.hdf5_fluid import create_fluid, create_boundary
from src.sql.entity.hdf5_iteration import Hdf5Iteration, load_hdf5_iterations
from src.progress import ProgressFunction, StatusHandler

from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, Union


@dataclass
class Simulation:
    simulation_name: str
    id: int = None
    config: Config = None
    hdf5_iterations: Dict[int, Hdf5Iteration] = None
    csv_iterations: Dict[int, CSVIteration] = None
    boundary: Boundary = None

    all: BulkCollector = None

    def exists(self, connection) -> (bool, Union[int, None]):
        return connection.exists(f"SELECT id FROM simulation WHERE simulation_name='{self.simulation_name}';")

    def insert(self, connection):
        (exists, _id) = connection.exists(f"SELECT id FROM simulation WHERE simulation_name='{self.simulation_name}';")

        if exists:
            self.id = _id
        else:
            self.id = connection.insert(f"""INSERT INTO simulation(simulation_name, created_at) VALUES (
                '{self.simulation_name}',
                '{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            );""")

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE simulation (
                    id integer PRIMARY KEY,
                    simulation_name text NOT NULL UNIQUE,
                    created_at text NOT NULL                   
                  );"""


def parse_simulation(connection, simulation_name: str, data_directory: str) -> Simulation:
    simulation = Simulation(simulation_name)

    (exists, simulation_id) = simulation.exists(connection)

    start_hdf5_iteration = 0
    start_csv_iteration = 0

    if exists:
        continue_process = input(f"A simulation with the name '{simulation_name}' already exists in the database. Do you want to add the data to the existing simulation? [Yes/No]\n")

        if continue_process.upper() == 'YES':
            print("Cleaning up data before continuing")
            start_hdf5_iteration, start_csv_iteration = connection.cleanup_before_continue(simulation_id)
            print("Finished cleaning up the data")
        else:
            replace = input(f"Do you want to overwrite it? [Yes/No]\n")

            if replace.upper() == 'YES':
                print("Removing previous simulation from database...")
                connection.remove_simulation(simulation_name)
                print("Removal successful")
            else:
                print("Okay, stopping program")
                exit(0)

    simulation.insert(connection)

    simulation.config = create_config(connection, simulation.id, data_directory)
    simulation.boundary = insert_boundary(connection, simulation, data_directory)

    simulation.hdf5_iterations = {}
    directories = list(filter(
        lambda d: int(d.split("/")[-2]) >= start_hdf5_iteration,
        sorted([d for d in glob(f"{data_directory}/hdf5/*/")])
    ))

    progress = ProgressFunction(len(directories), f"{simulation_name} | Inserting HDF5 iterations")
    for directory in directories:
        progress.run(insert_hdf5_iteration, connection, simulation, directory)

    simulation.csv_iterations = {}
    iterations = list(filter(
        lambda i: int(i) >= start_csv_iteration,
        sorted(list(set([d.split(".")[-2] for d in glob(f"{data_directory}/csv/*")])))
    ))

    progress = ProgressFunction(len(iterations), f"{simulation_name} | Inserting CSV iterations")
    for iteration in iterations:
        progress.run(insert_csv_iteration, connection, simulation, data_directory, iteration)

    return simulation


def insert_boundary(connection, simulation, data_directory) -> Boundary:
    directory = sorted([d for d in glob(f"{data_directory}/hdf5/*/")])[0]

    return create_boundary(connection, directory, simulation.id, simulation.config)


def insert_hdf5_iteration(connection, simulation, directory):
    iteration_int = int(directory.split("/")[-2])
    iteration_object = Hdf5Iteration(connection, iteration_int)
    iteration_object.insert(connection, simulation.id)
    simulation.hdf5_iterations[iteration_int] = iteration_object

    cell_positions = create_hdf5_cells(connection, iteration_object.id, directory, prefix="PLT")
    cell_positions += create_hdf5_cells(connection, iteration_object.id, directory, prefix="RBC")

    create_fluid(connection, iteration_object.id, directory, simulation.id, simulation.config, cell_positions, simulation.boundary)


def insert_csv_iteration(connection, simulation, data_directory, iteration):
    iteration_int = int(iteration)
    iteration_object = CSVIteration(connection, iteration_int)
    iteration_object.insert(connection, simulation.id)
    simulation.csv_iterations[iteration_int] = iteration_object

    create_csv_cells(
        connection,
        iteration_object.id,
        f"{data_directory}/csv/PLT.{iteration}.csv",
        "PLT"
    )
    create_csv_cells(
        connection,
        iteration_object.id,
        f"{data_directory}/csv/RBC.{iteration}.csv",
        "RBC"
    )


def load_simulation(connection, simulation_name: str, status: StatusHandler) -> Simulation:
    try:
        simulation = Simulation(
            *connection.select_one(f"""
                SELECT simulation_name, id 
                FROM simulation 
                WHERE simulation_name='{simulation_name}'
            ;""")
        )
    except TypeError:
        print(f'Unable to find a simulation with the name "{simulation_name}"')
        exit(1)

    simulation.config = load_config(connection, simulation.id)
    simulation.hdf5_iterations = load_hdf5_iterations(connection, simulation.id)
    simulation.csv_iterations = load_csv_iterations(connection, simulation.id)
    simulation.boundary = load_boundary(connection, simulation.id)
    simulation.all = BulkCollector(connection, simulation.id, simulation.hdf5_iterations, simulation.csv_iterations, status)

    return simulation
