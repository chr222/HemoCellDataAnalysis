import pytest
import os

from progress import StatusHandler
from src.sql.entity.simulation import parse_simulation, load_simulation
from src.sql.connection import Connection

simulation_name = 'test_project'
data_directory = 'test_data'
database_path = '/tmp/test.db'
matrix_path = '/tmp/test_matrices'


@pytest.fixture
def connection():
    if os.path.exists(database_path):
        os.remove(database_path)

    return Connection(database_path, matrix_path)


@pytest.fixture
def simulation(connection):
    parse_simulation(connection, simulation_name, data_directory)
    return load_simulation(connection, simulation_name, StatusHandler())


def test_load_fluid(simulation):
    assert len(simulation.all.fluid_velocities()) == 3
    assert len(simulation.all.fluid_shear_stress()) == 3
    assert len(simulation.all.fluid_shear_rate()) == 3
    assert len(simulation.all.hematocrits()) == 3


def test_load_red_blood_cells(simulation):
    assert len(simulation.all.red_blood_cell_positions()) == 0
    assert len(simulation.all.red_blood_cell_velocities()) == 0
    assert len(simulation.all.red_blood_cell_counts()) == 3


def test_load_platelets(simulation):
    assert len(simulation.all.platelet_positions()) == 3
    assert len(simulation.all.platelet_velocities()) == 3
    assert len(simulation.all.platelet_counts()) == 3
