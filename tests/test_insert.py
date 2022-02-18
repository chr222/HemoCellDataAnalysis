import numpy as np
from bs4 import BeautifulSoup
import os
from pathlib import Path
import pytest

import params
from sql.entity.block import get_domain_info
from src.sql.entity.simulation import Simulation, insert_boundary, insert_hdf5_iteration, insert_csv_iteration, \
    parse_simulation
from src.sql.entity.config import create_config, Config
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
def empty_simulation(connection):
    simulation = Simulation(simulation_name=simulation_name)
    simulation.insert(connection)

    return simulation


@pytest.fixture
def simulation(connection):
    return parse_simulation(connection, simulation_name, data_directory)


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


def test_insert_config(connection, empty_config):
    assert empty_config.id is not None


def test_get_domain_info(connection, empty_simulation, empty_config):
    blocks_data, domain = get_domain_info(data_directory)

    assert np.any(domain > 0)

    for atomic_block in blocks_data.keys():
        assert blocks_data[atomic_block] is not None


def test_insert_boundary(connection, empty_simulation, config):
    empty_simulation.config = config
    boundary = insert_boundary(connection, empty_simulation, data_directory)

    assert boundary.id is not None


def test_insert_hdf5_iterations(connection, empty_simulation, config):
    empty_simulation.config = config
    empty_simulation.boundary = insert_boundary(connection, empty_simulation, data_directory)

    empty_simulation.hdf5_iterations = {}
    directories = sorted(list(data_directory.glob("hdf5/*")))

    for directory in directories:
        insert_hdf5_iteration(connection, empty_simulation, directory)

        iteration_int = int(directory.name)
        assert empty_simulation.hdf5_iterations[iteration_int].id is not None


def test_insert_csv_iterations(connection, empty_simulation, config):
    empty_simulation.config = config
    empty_simulation.boundary = insert_boundary(connection, empty_simulation, data_directory)

    empty_simulation.csv_iterations = {}
    iterations = sorted(list(set([d.name.split(".")[-2] for d in data_directory.glob("csv/*")])))

    for iteration in iterations:
        insert_csv_iteration(connection, empty_simulation, data_directory, iteration)

        assert empty_simulation.csv_iterations[int(iteration)].id is not None


def test_insert_full_simulation(connection):
    simulation = parse_simulation(connection, simulation_name, data_directory)
    assert simulation.id is not None
