from __future__ import annotations

import sys

import params
from src.sql.entity import parent, exclude, Entity
from src.linalg import Vector3Int

from dataclasses import dataclass
from glob import glob
import h5py
import inspect
from math import ceil
import numpy as np
from typing import List, Annotated, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql.connection import Connection
    from src.sql.entity.boundary import Boundary
    from src.sql.entity.config import Config


@dataclass
class HDF5Fluid(Entity):
    iteration_id: Annotated[int, parent('hdf5_iteration', 'id')]
    _density: np.ndarray = None
    _force: np.ndarray = None
    _shear_rate: np.ndarray = None
    _shear_stress: np.ndarray = None
    _velocity: np.ndarray = None
    _hematocrit: np.ndarray = None
    connection: Annotated[Any, exclude] = None

    def _get_value(self) -> np.ndarray:
        # The name of the parent function is the column it should query
        column = '_' + inspect.stack()[1].function
        return self.connection.select_one(
            f"SELECT {column} FROM hdf5_fluid WHERE iteration_id=?;",
            self.iteration_id
        )[0]

    @property
    def density(self) -> np.ndarray:
        return self._get_value()

    @property
    def force(self) -> Vector3Matrix:
        return Vector3Matrix(self._get_value())

    @property
    def shear_rate(self) -> Tensor9Matrix:
        return Tensor9Matrix(self._get_value())

    @property
    def shear_stress(self) -> Tensor6Matrix:
        return Tensor6Matrix(self._get_value())

    @property
    def velocity(self) -> Vector3Matrix:
        return Vector3Matrix(self._get_value())

    @property
    def hematocrit(self) -> np.ndarray:
        return self._get_value()


def create_fluid(connection: "Connection", iteration_id: int, directory: str, config: "Config", cells: List[np.ndarray] = None, boundary: "Boundary" = None):
    data = {k: None for (_, k) in params.HDF5_FLUID_FIELDS}

    files = sorted([f for f in glob(directory + f"Fluid.*.p.*.h5")])
    for i, file_path in enumerate(files):
        # Get atomic block from file path
        atomic_block = file_path.split(".")[-2]

        hdf5_data = {}
        with h5py.File(file_path, 'r') as f:
            for hdf5_name, entity_name in params.HDF5_FLUID_FIELDS:
                try:
                    hdf5_data[entity_name] = np.array(f[hdf5_name])
                except KeyError:
                    print(f"Could not find '{hdf5_name}' in the HDF5 fluid data", file=sys.stderr)

        # Initialize data
        if i == 0:
            for k, hdf5_dataset in hdf5_data.items():
                vector_size = hdf5_dataset.shape[3]

                # Handle as scalar
                if vector_size == 1:
                    data[k] = np.zeros((config.ny, config.nx, config.nz))
                else:
                    data[k] = np.zeros((config.ny, config.nx, config.nz, vector_size))

        size: Vector3Int = config.blocks[atomic_block].size
        offset: Vector3Int = config.blocks[atomic_block].offset
        for z in range(1, size.z + 1):
            for y in range(1, size.y + 1):
                for x in range(1, size.x + 1):
                    # Coordinates in complete container
                    real_x = x - 1 + offset.x
                    real_y = y - 1 + offset.y
                    real_z = z - 1 + offset.z

                    for k, hdf5_dataset in hdf5_data.items():
                        vector_size = hdf5_dataset.shape[3]

                        # Handle as scalar
                        if vector_size == 1:
                            data[k][real_y][real_x][real_z] = hdf5_dataset[z][y][x][0]
                        else:
                            data[k][real_y][real_x][real_z] = hdf5_dataset[z][y][x]

    fluid = HDF5Fluid.from_dict(
        iteration_id=iteration_id,
        connection=connection,
        _hematocrit=create_hematocrit(config, cells, boundary) if cells is not None else None,
        **data
    )
    fluid.insert(connection)


def create_hematocrit(config: "Config", cells: List[np.ndarray], boundary: "Boundary" or None) -> np.ndarray:
    hematocrit = np.zeros((config.ny, config.nx, config.nz))

    for cell in cells:
        # Positions of the triangles of each blood cell
        positions = np.divide(cell, config.dx)

        # The center of each blood cell
        center = np.mean(positions, axis=0)

        for position in positions:
            length = np.sum(np.power(center - position, 2))
            steps = ceil(length)
            remainder = abs(length)

            # The lattice cells between the center and each triangle are set to the percentage that is covered by
            # the blood cell
            for x, y, z in np.transpose([
                np.linspace(center[0], position[0], steps, dtype=int),
                np.linspace(center[1], position[1], steps, dtype=int),
                np.linspace(center[2], position[2], steps, dtype=int)
            ]):
                if remainder < 0:
                    break

                # Keep value between 0 and 1
                new_value = max(0.0, min(hematocrit[y][x][z] + remainder, 1.0))
                hematocrit[y][x][z] = new_value
                remainder -= 1.0

    # Set fluid cells outside and on the border of the container to NaN, so they can be excluded
    # from further calculations
    if boundary is not None:
        hematocrit[np.where(boundary.boundary_map == 1)] = np.nan

    return hematocrit


class Vector3Matrix(np.ndarray):
    def __new__(cls, array: np.ndarray):
        return np.asarray(array).view(cls)

    def __array_wrap__(self, obj):
        return np.array(obj)

    def exclude_borders(self, boundary: "Boundary" or None) -> Vector3Matrix:
        """
        Set the cells that are boundaries to NaN
        """

        if boundary is None:
            return self

        self[boundary.boundaries] = np.full(3, np.nan)

        return self

    @property
    def x(self) -> np.ndarray:
        return self[:, :, :, 0]

    @property
    def y(self) -> np.ndarray:
        return self[:, :, :, 1]

    @property
    def z(self) -> np.ndarray:
        return self[:, :, :, 2]

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self, 2), axis=3))


class Tensor9Matrix(np.ndarray):
    def __new__(cls, array: np.ndarray):
        return np.asarray(array).view(cls)

    def __array_wrap__(self, obj):
        return np.array(obj)

    def exclude_borders(self, boundary: "Boundary" or None) -> Tensor9Matrix:
        """
        Set the cells that are boundaries to NaN
        """

        if boundary is None:
            return self

        matrix = np.copy(self)
        matrix[boundary.boundaries] = np.full(9, np.nan)

        return Tensor9Matrix(matrix)

    @property
    def x_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 1], 2) + np.power(self[:, :, :, 2], 2))

    @property
    def y_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 3], 2) + np.power(self[:, :, :, 5], 2))

    @property
    def z_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 6], 2) + np.power(self[:, :, :, 7], 2))

    @property
    def x_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, 0:3], 2), axis=3))

    @property
    def y_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, 3:6], 2), axis=3))

    @property
    def z_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, 6:9], 2), axis=3))

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self, 2), axis=3))


class Tensor6Matrix(np.ndarray):
    def __new__(cls, array: np.ndarray):
        return np.asarray(array).view(cls)

    def __array_wrap__(self, obj):
        return np.array(obj)

    def exclude_borders(self, boundary: "Boundary" or None) -> Tensor6Matrix:
        """
        Set the cells that are boundaries to NaN
        """

        if boundary is None:
            return self

        matrix = np.copy(self)
        matrix[boundary.boundaries] = np.full(6, np.nan)

        return Tensor6Matrix(matrix)

    @property
    def x_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 1], 2) + np.power(self[:, :, :, 2], 2))

    @property
    def y_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 1], 2) + np.power(self[:, :, :, 4], 2))

    @property
    def z_plane(self) -> np.ndarray:
        return np.sqrt(np.power(self[:, :, :, 2], 2) + np.power(self[:, :, :, 4], 2))

    @property
    def x_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, (0, 1, 2)], 2), axis=3))

    @property
    def y_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, (1, 3, 4)], 2), axis=3))

    @property
    def z_face(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self[:, :, :, (2, 4, 5)], 2), axis=3))

    @property
    def magnitude(self) -> np.ndarray:
        return np.sqrt(np.sum(np.power(self, 2), axis=3))
