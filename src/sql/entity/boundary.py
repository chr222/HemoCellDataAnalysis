from dataclasses import dataclass
from glob import glob
import h5py
import numpy as np
from scipy.signal import fftconvolve
from typing import Annotated, TYPE_CHECKING

from src.linalg import Vector3Int
from src.sql.entity.config import Config
from src.sql.entity import parent, Entity, EntityNotFoundException

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


def create_boundary(connection, directory: str, simulation_id: int, config: Config) -> Boundary or None:
    if Boundary.is_already_created(connection, simulation_id):
        return load_boundary(connection, simulation_id)

    boundary_map = np.zeros((config.ny, config.nx, config.nz))

    files = sorted([f for f in glob(directory + f"Fluid.*.p.*.h5")])
    for file_path in files:
        # Get atomic block from file path
        atomic_block = file_path.split(".")[-2]

        with h5py.File(file_path, 'r') as f:
            try:
                hdf5_is_boundary = np.array(f['Boundary'])
            except KeyError:
                print("Could not find the Boundary map in the HDF5 fluid dataset")
                return None

        size: Vector3Int = config.blocks_data[atomic_block].size
        offset: Vector3Int = config.blocks_data[atomic_block].offset
        for z in range(1, size.z + 1):
            for y in range(1, size.y + 1):
                for x in range(1, size.x + 1):
                    # Coordinates in complete container
                    real_x = x - 1 + offset.x
                    real_y = y - 1 + offset.y
                    real_z = z - 1 + offset.z

                    boundary_map[real_y][real_x][real_z] = hdf5_is_boundary[z][y][x][0]

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
