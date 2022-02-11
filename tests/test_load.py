import json
import sys
from glob import glob
from typing import Dict

import pytest
import os

from linalg import Vector3Int
from progress import StatusHandler
from sql.entity.block import Block
from sql.entity.boundary import load_boundary
from sql.entity.csv_iteration import load_csv_iterations
from sql.entity.hdf5_iteration import Hdf5Iteration, load_hdf5_iterations
from src.sql.entity.simulation import Simulation, insert_boundary, insert_hdf5_iteration, insert_csv_iteration, \
    parse_simulation, load_simulation
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
    with open(f"{data_directory}/log/logfile", "r") as f:
        data = "".join([line for line in f.readlines()])
        result = connection.find_json_string("ConfigParams", data)
        params = json.loads(result)
        config = Config.from_dict(simulation_id=empty_simulation.id, **params)
        config.insert(connection)

    return config


@pytest.fixture
def config(connection, empty_simulation):
    return create_config(connection, empty_simulation.id, data_directory)


def test_load_config_without_blocks(connection, empty_simulation, empty_config):
    found_config = load_config(connection, empty_simulation.id)

    assert found_config.blocks_data is None
    assert found_config.id is not None


def test_load_config_with_blocks(connection, empty_simulation, config):
    found_config = load_config(connection, empty_simulation.id)

    assert found_config.blocks_data is not None
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