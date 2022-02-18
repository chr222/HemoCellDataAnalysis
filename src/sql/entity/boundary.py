import sys
from dataclasses import dataclass
import h5py
import numpy as np
from pathlib import Path
from scipy.signal import fftconvolve
from typing import Annotated, TYPE_CHECKING

from src.linalg import Vector3Int
from src.sql.entity import parent, Entity, EntityNotFoundException
from src.sql.entity.config import Config
from src.sql.entity.hdf5_fluid import parse_hdf5_fluid_data

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Boundary(Entity):
    simulation_id: Annotated[int, parent('simulation', 'id')]
    boundary_map: np.ndarray

    @staticmethod
    def is_already_created(connection: "Connection", simulation_id: int) -> bool:
        return connection.exists(f"SELECT id FROM boundary WHERE simulation_id=?;", simulation_id)[0]

    @property
    def boundaries(self) -> np.ndarray:
        kernel = np.ones((3, 3, 3))
        boundaries = np.minimum(fftconvolve(self.boundary_map, kernel, mode='same'), 1.0)

        return np.where(boundaries == 1)


def get_hdf5_file(file_path: str):
    with h5py.File(file_path, 'r') as f:
        return np.array(f['Boundary'])


def place_boundary(config, atomic_block, boundary_map, hdf5_is_boundary):
    size: Vector3Int = config.blocks[atomic_block].size
    offset: Vector3Int = config.blocks[atomic_block].offset
    data = np.moveaxis(hdf5_is_boundary, 0, 2)[1:size.y + 1, 1:size.x + 1, 1:size.z + 1]

    start_x = offset.x
    end_x = size.x + offset.x

    start_y = offset.y
    end_y = size.y + offset.y

    start_z = offset.z
    end_z = size.z + offset.z

    boundary_map[start_y:end_y, start_x:end_x, start_z:end_z] = data[:, :, :, 0]


def create_boundary(connection, directory: Path, simulation_id: int, config: Config) -> Boundary or None:
    if Boundary.is_already_created(connection, simulation_id):
        return load_boundary(connection, simulation_id)

    boundary_map = np.zeros((config.ny, config.nx, config.nz))

    files = sorted(list(directory.glob("Fluid.*.p.*.h5")))
    for file_path in files:
        # Get atomic block from file path
        atomic_block = file_path.name.split(".")[-2]

        try:
            with h5py.File(file_path, 'r') as f:
                hdf5_is_boundary = np.array(f['Boundary'])
        except KeyError:
            print("Could not find the Boundary map in the HDF5 fluid dataset", file=sys.stderr)
            return None

        parse_hdf5_fluid_data(config, atomic_block, hdf5_is_boundary, output=boundary_map)

    boundary = Boundary(
        simulation_id=simulation_id,
        boundary_map=boundary_map
    )
    boundary.insert(connection)

    return boundary


def load_boundary(connection: "Connection", simulation_id: int) -> Boundary or None:
    try:
        return Boundary.load(connection, simulation_id)
    except EntityNotFoundException:
        return None
