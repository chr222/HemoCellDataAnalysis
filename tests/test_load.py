from bs4 import BeautifulSoup
import pytest
import os

import params
from progress import StatusHandler
from src.sql.entity.boundary import load_boundary
from src.sql.entity.csv_iteration import load_csv_iterations
from src.sql.entity.hdf5_iteration import load_hdf5_iterations
from src.sql.entity.simulation import Simulation, parse_simulation, load_simulation
from src.sql.entity.config import create_config, Config, load_config
from src.sql.connection import Connection

simulation_name = 'test_project'
data_directory = 'test_data'
database_path = '/tmp/test.db'


@pytest.fixture
def connection():
    if os.path.exists(database_path):
        os.remove(database_path)

    return Connection(database_path)


@pytest.fixture
def empty_simulation(connection):
    simulation = Simulation(simulation_name=simulation_name)
    simulation.insert(connection)

    return simulation


@pytest.fixture
def simulation(connection):
    return parse_simulation(connection, simulation_name, data_directory)


@pytest.fixture
def loaded_simulation(connection):
    parse_simulation(connection, simulation_name, data_directory)
    return load_simulation(connection, simulation_name, StatusHandler())


@pytest.fixture
def empty_config(connection, empty_simulation):
    with open(f"{data_directory}/config.xml", 'r') as f:
        data = BeautifulSoup(f.read(), 'xml')

    config_params = {}
    for config_field, entity_field in [('dx', 'dx'), ('dt', 'dt')] + params.CONFIG_FIELDS:
        tag = data.find(config_field)

        if tag is None:
            continue

        value_type = Config.get_property_type(entity_field)
        config_params[entity_field] = value_type(tag.text.strip())

    config = Config(simulation_id=empty_simulation.id, **config_params)
    config.insert(connection)

    return config


@pytest.fixture
def config(connection, empty_simulation):
    return create_config(connection, empty_simulation.id, data_directory)


def test_load_config(connection, empty_simulation, empty_config):
    found_config = load_config(connection, empty_simulation.id)

    assert found_config.id is not None


def test_load_no_hdf5_iterations(connection, empty_simulation):
    found_hdf5_iterations = load_hdf5_iterations(connection, empty_simulation.id)

    assert len(found_hdf5_iterations.keys()) == 0


def test_load_hdf5_iterations(connection, simulation):
    found_hdf5_iterations = load_hdf5_iterations(connection, simulation.id)

    assert len(found_hdf5_iterations.keys()) > 0

    for _, iteration in found_hdf5_iterations.items():
        assert iteration.id is not None


def test_load_no_csv_iterations(connection, empty_simulation):
    found_csv_iterations = load_csv_iterations(connection, empty_simulation.id)

    assert len(found_csv_iterations.keys()) == 0


def test_load_csv_iterations(connection, simulation):
    found_csv_iterations = load_csv_iterations(connection, simulation.id)

    assert len(found_csv_iterations.keys()) > 0

    for _, iteration in found_csv_iterations.items():
        assert iteration.id is not None


def test_load_no_boundary(connection, empty_simulation):
    found_boundary = load_boundary(connection, empty_simulation.id)

    assert found_boundary is None


def test_load_boundary(connection, simulation):
    found_boundary = load_boundary(connection, simulation.id)

    assert found_boundary.id is not None