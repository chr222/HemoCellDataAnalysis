import sys

from src.sql.entity import Entity, exclude, unique
from src.sql.entity.boundary import Boundary, load_boundary, create_boundary
from src.sql.entity.bulk_collector import BulkCollector
from src.sql.entity.config import create_config, load_config, Config
from src.sql.entity.csv_cell import create_csv_cells
from src.sql.entity.csv_iteration import CSVIteration, load_csv_iterations
from src.sql.entity.hdf5_cell import create_hdf5_cells
from src.sql.entity.hdf5_fluid import create_fluid
from src.sql.entity.hdf5_iteration import Hdf5Iteration, load_hdf5_iterations
from src.progress import ProgressFunction, StatusHandler

from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Dict, Union, Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Simulation(Entity):
    simulation_name: Annotated[str, unique]
    created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config: Annotated[Config, exclude] = None
    hdf5_iterations: Annotated[Dict[int, Hdf5Iteration], exclude] = None
    csv_iterations: Annotated[Dict[int, CSVIteration], exclude] = None
    boundary: Annotated[Boundary, exclude] or None = None
    all: Annotated[BulkCollector, exclude] = None

    def exists(self, connection: "Connection") -> (bool, Union[int, None]):
        return connection.exists(
            "SELECT id FROM simulation WHERE simulation_name=?;",
            self.simulation_name
        )

    @classmethod
    def load(cls, connection: "Connection", simulation_name: str) -> "Simulation":
        params = connection.select_one(
            "SELECT simulation_name, id FROM simulation WHERE simulation_name=?;",
            simulation_name
        )

        return Simulation.from_dict(
            id=params[1],
            simulation_name=params[0]
        )


def parse_simulation(connection: "Connection", simulation_name: str, data_directory: str, config_path: str = None) -> Simulation:
    simulation = Simulation(simulation_name=simulation_name)

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

    simulation.config = create_config(connection, simulation.id, data_directory, config_path)

    if simulation.config.blocks is not None:
        try:
            simulation.boundary = insert_boundary(connection, simulation, data_directory)
        except IndexError:
            print("Could not parse the HDF5 boundary map since the atomic block info is incorrect.", file=sys.stderr)

        simulation.hdf5_iterations = {}

        # Get remaining directories
        directories = list(filter(
            lambda d: int(d.split("/")[-2]) >= start_hdf5_iteration,
            sorted([d for d in glob(f"{data_directory}/hdf5/*/")])
        ))

        # Insert HDF5 iterations data into the database
        try:
            progress = ProgressFunction(len(directories), f"{simulation_name} | Inserting HDF5 iterations")
            for directory in directories:
                progress.run(insert_hdf5_iteration, connection, simulation, directory)
        except IndexError:
            print("\nCould not parse the HDF5 fluid data since the atomic blocks data is incorrect.\n", file=sys.stderr)
    else:
        print(f"Unable to parse HDF5 files due to the blocks info missing", file=sys.stderr)

    simulation.csv_iterations = {}

    # Get remaining csv iterations
    iterations = list(filter(
        lambda i: int(i) >= start_csv_iteration,
        sorted(list(set([d.split(".")[-2] for d in glob(f"{data_directory}/csv/*")])))
    ))

    # Insert the CSV iterations data into the database
    progress = ProgressFunction(len(iterations), f"{simulation_name} | Inserting CSV iterations")
    for iteration in iterations:
        progress.run(insert_csv_iteration, connection, simulation, data_directory, iteration)

    return simulation


def insert_boundary(connection: "Connection", simulation: Simulation, data_directory: str) -> Boundary or None:
    # The boundary is always the same, so it is only imported by parsing the first iteration
    directory = sorted([d for d in glob(f"{data_directory}/hdf5/*/")])[0]

    return create_boundary(connection, directory, simulation.id, simulation.config)


def insert_hdf5_iteration(connection: "Connection", simulation: Simulation, directory: str):
    iteration_int = int(directory.split("/")[-2])
    iteration_object = Hdf5Iteration(
        simulation_id=simulation.id,
        iteration=iteration_int,
        connection=connection
    )
    iteration_object.insert(connection)
    simulation.hdf5_iterations[iteration_int] = iteration_object

    plt_positions = create_hdf5_cells(connection, iteration_object.id, directory, prefix="PLT")
    rbc_positions = create_hdf5_cells(connection, iteration_object.id, directory, prefix="RBC")

    if plt_positions is not None and rbc_positions is not None:
        cell_positions = plt_positions + rbc_positions
    else:
        if len(list(simulation.hdf5_iterations.keys())) <= 1:
            print("The Cell Id field is missing from the HDF5 cell data. As a result the cell data cannot be parsed.", file=sys.stderr)

        cell_positions = None

    create_fluid(connection, iteration_object.id, directory, simulation.config, cell_positions, simulation.boundary)


def insert_csv_iteration(connection: "Connection", simulation: Simulation, data_directory: str, iteration: str):
    iteration_int = int(iteration)
    iteration_object = CSVIteration.from_dict(
        simulation_id=simulation.id,
        iteration=iteration_int,
        connection=connection
    )
    iteration_object.insert(connection)
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


def load_simulation(connection: "Connection", simulation_name: str, status: StatusHandler) -> Simulation:
    try:
        simulation = Simulation.load(connection, simulation_name)
    except TypeError:
        print(f'Unable to find a simulation with the name "{simulation_name}"', file=sys.stderr)
        raise exit(1)

    simulation.config = load_config(connection, simulation.id)
    simulation.hdf5_iterations = load_hdf5_iterations(connection, simulation.id)
    simulation.csv_iterations = load_csv_iterations(connection, simulation.id)
    simulation.boundary = load_boundary(connection, simulation.id)
    simulation.all = BulkCollector(connection, simulation.id, simulation.hdf5_iterations, simulation.csv_iterations, status)

    return simulation
