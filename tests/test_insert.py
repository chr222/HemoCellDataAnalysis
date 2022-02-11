import json
import sys
from glob import glob
from typing import Dict

import pytest
import os

from linalg import Vector3Int
from sql.entity.block import Block
from sql.entity.hdf5_iteration import Hdf5Iteration
from src.sql.entity.simulation import Simulation, insert_boundary, insert_hdf5_iteration, insert_csv_iteration, \
    parse_simulation
from src.sql.entity.config import create_config, Config
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


def test_insert_config(connection, empty_config):
    assert empty_config.id is not None


def test_insert_blocks(connection, empty_simulation, empty_config):
    with open(f"{data_directory}/log/logfile", "r") as f:
        data = "".join([line for line in f.readlines()])
        result = connection.find_json_string("ConfigParams", data)

        params = json.loads(result)

        blocks_data = {}
        for atomic_block, block in params['blocks'].items():
            blocks_data[atomic_block] = Block(
                config_id=empty_config.id,
                atomic_block=atomic_block,
                size=Vector3Int(*block['size']),
                offset=Vector3Int(*block['offset'])
            )
            blocks_data[atomic_block].insert(connection)
            assert blocks_data[atomic_block].id is not None


def test_insert_boundary(connection, empty_simulation, config):
    empty_simulation.config = config
    boundary = insert_boundary(connection, empty_simulation, data_directory)

    assert boundary.id is not None


def test_insert_hdf5_iterations(connection, empty_simulation, config):
    empty_simulation.config = config
    empty_simulation.boundary = insert_boundary(connection, empty_simulation, data_directory)

    empty_simulation.hdf5_iterations = {}
    directories = sorted([d for d in glob(f"{data_directory}/hdf5/*/")])

    for directory in directories:
        insert_hdf5_iteration(connection, empty_simulation, directory)

        iteration_int = int(directory.split("/")[-2])
        assert empty_simulation.hdf5_iterations[iteration_int].id is not None


def test_insert_csv_iterations(connection, empty_simulation, config):
    empty_simulation.config = config
    empty_simulation.boundary = insert_boundary(connection, empty_simulation, data_directory)

    empty_simulation.csv_iterations = {}
    iterations = sorted(list(set([d.split(".")[-2] for d in glob(f"{data_directory}/csv/*")])))

    for iteration in iterations:
        insert_csv_iteration(connection, empty_simulation, data_directory, iteration)

        assert empty_simulation.csv_iterations[int(iteration)].id is not None


def test_insert_full_simulation(connection):
    simulation = parse_simulation(connection, simulation_name, data_directory)
    assert simulation.id is not None
