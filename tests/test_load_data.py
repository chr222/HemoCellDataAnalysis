import pytest
import os
from pathlib import Path

from progress import StatusHandler
from src.sql.entity.simulation import parse_simulation, load_simulation
from src.sql.connection import Connection

simulation_name = 'test_project'
data_directory = Path('test_data')
database_path = Path('/tmp/test.db')
matrix_path = Path('/tmp/test_matrices')


@pytest.fixture
def connection():
    if os.path.exists(database_path):
        os.remove(database_path)

    return Connection(database_path, matrix_path)


@pytest.fixture
def simulation(connection):
    parse_simulation(connection, simulation_name, data_directory)
    return load_simulation(connection, simulation_name, StatusHandler())


def test_load_fluid(connection, simulation):
    fluid = simulation.hdf5_iterations[0].fluid

    assert fluid.density is not None
    assert fluid.force is not None
    assert fluid.shear_rate is not None
    assert fluid.shear_stress is not None
    assert fluid.velocity is not None
    assert fluid.hematocrit is not None


def test_load_hdf5_red_blood_cells(connection, simulation):
    rbc = simulation.hdf5_iterations[0].red_blood_cells

    assert rbc.count is not None
    assert rbc.area_force is not None
    assert rbc.bending_force is not None
    assert rbc.inner_link_force is not None
    assert rbc.link_force is not None
    assert rbc.position is not None
    assert rbc.repulsion_force is not None
    assert rbc.total_force is not None
    assert rbc.velocity is not None
    assert rbc.viscous_force is not None
    assert rbc.volume_force is not None


def test_load_hdf5_platelets(connection, simulation):
    plt = simulation.hdf5_iterations[0].platelets

    assert plt.count is not None
    assert plt.area_force is not None
    assert plt.bending_force is not None
    assert plt.inner_link_force is not None
    assert plt.link_force is not None
    assert plt.position is not None
    assert plt.repulsion_force is not None
    assert plt.total_force is not None
    assert plt.velocity is not None
    assert plt.viscous_force is not None
    assert plt.volume_force is not None


def test_load_csv_red_blood_cells(connection, simulation):
    rbc = simulation.csv_iterations[0].red_blood_cells

    assert rbc.position is not None
    assert rbc.velocity is not None
    assert rbc.atomic_block is not None
    assert rbc.area is not None
    assert rbc.volume is not None


def test_load_csv_platelets(connection, simulation):
    plt = simulation.csv_iterations[0].platelets

    assert plt.position is not None
    assert plt.velocity is not None
    assert plt.atomic_block is not None
    assert plt.area is not None
    assert plt.volume is not None
