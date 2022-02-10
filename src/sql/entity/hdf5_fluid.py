from __future__ import annotations

from src.sql.entity.boundary import Boundary, load_boundary
from src.sql.entity.config import Config
from src.linalg import Vector3Int

from dataclasses import dataclass
from glob import glob
import h5py
import inspect
from math import ceil
import numpy as np
from typing import Any, List


class Vector3Matrix(np.ndarray):
    def __new__(cls, array: np.ndarray):
        return np.asarray(array).view(cls)

    def __array_wrap__(self, obj):
        return np.array(obj)

    def exclude_borders(self, boundary: Boundary) -> Vector3Matrix:
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

    def exclude_borders(self, boundary: Boundary) -> Tensor9Matrix:
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

    def exclude_borders(self, boundary: Boundary) -> Tensor6Matrix:
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


@dataclass
class Fluid:
    connection: Any
    iteration_id: int
    id: int = None
    
    def insert(self, *args):
        self.id = self.connection.insert(
            f"""
            INSERT INTO fluid (iteration_id, density, force, shear_rate, shear_stress, velocity, hematocrit) 
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (self.iteration_id, *args)
        )

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE fluid (
                    id integer PRIMARY KEY,
                    iteration_id integer,
                    density array,
                    force array,
                    shear_rate array,
                    shear_stress array,
                    velocity array,
                    hematocrit array,
                    FOREIGN KEY (iteration_id) REFERENCES hdf5_iteration (id) ON DELETE CASCADE     
                  );"""

    def _get_value(self) -> np.ndarray:
        # The name of the parent function is the column it should query
        column = inspect.stack()[1].function
        return self.connection.select_one(f"""SELECT {column} FROM fluid WHERE iteration_id='{self.iteration_id}';""")[0]

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


def create_fluid(connection, iteration_id: int, directory: str, simulation_id: int, config: Config, cells: List[np.ndarray], boundary: Boundary):
    density = np.zeros((config.ny, config.nx, config.nz))
    force = np.zeros((config.ny, config.nx, config.nz, 3))
    shear_rate = np.zeros((config.ny, config.nx, config.nz, 9))
    shear_stress = np.zeros((config.ny, config.nx, config.nz, 6))
    velocity = np.zeros((config.ny, config.nx, config.nz, 3))

    files = sorted([f for f in glob(directory + f"Fluid.*.p.*.h5")])
    for file_path in files:
        # Get atomic block from file path
        atomic_block = file_path.split(".")[-2]

        with h5py.File(file_path, 'r') as f:
            hdf5_density = np.array(f['Density'])
            hdf5_force = np.array(f['Force'])
            hdf5_shear_rate = np.array(f['ShearRate'])
            hdf5_shear_stress = np.array(f['ShearStress'])
            hdf5_velocity = np.array(f['Velocity'])

        size: Vector3Int = config.blocks[atomic_block].size
        offset: Vector3Int = config.blocks[atomic_block].offset
        for z in range(1, size.z + 1):
            for y in range(1, size.y + 1):
                for x in range(1, size.x + 1):
                    # Coordinates in complete container
                    real_x = x - 1 + offset.x
                    real_y = y - 1 + offset.y
                    real_z = z - 1 + offset.z

                    density[real_y][real_x][real_z] = hdf5_density[z][y][x][0]
                    force[real_y][real_x][real_z] = hdf5_force[z][y][x]
                    shear_rate[real_y][real_x][real_z] = hdf5_shear_rate[z][y][x]
                    shear_stress[real_y][real_x][real_z] = hdf5_shear_stress[z][y][x]
                    velocity[real_y][real_x][real_z] = hdf5_velocity[z][y][x]

    fluid = Fluid(connection, iteration_id)
    fluid.insert(density, force, shear_rate, shear_stress, velocity, create_hematocrit(config, cells, boundary.boundary_map))


def create_boundary(connection, directory: str, simulation_id: int, config: Config) -> Boundary:
    if Boundary.is_already_created(connection, simulation_id):
        return load_boundary(connection, simulation_id)

    boundary_map = np.zeros((config.ny, config.nx, config.nz))

    files = sorted([f for f in glob(directory + f"Fluid.*.p.*.h5")])
    for file_path in files:
        # Get atomic block from file path
        atomic_block = file_path.split(".")[-2]

        with h5py.File(file_path, 'r') as f:
            hdf5_is_boundary = np.array(f['Boundary'])

        size: Vector3Int = config.blocks[atomic_block].size
        offset: Vector3Int = config.blocks[atomic_block].offset
        for z in range(1, size.z + 1):
            for y in range(1, size.y + 1):
                for x in range(1, size.x + 1):
                    # Coordinates in complete container
                    real_x = x - 1 + offset.x
                    real_y = y - 1 + offset.y
                    real_z = z - 1 + offset.z

                    boundary_map[real_y][real_x][real_z] = hdf5_is_boundary[z][y][x][0]

    boundary = Boundary(boundary_map)
    boundary.insert(connection, simulation_id)

    return boundary


def create_hematocrit(config: Config, cells: List[np.ndarray], boundary_map: np.ndarray) -> np.ndarray:
    hematocrit = np.zeros((config.ny, config.nx, config.nz))

    for cell in cells:
        positions = np.divide(cell, config.dx)
        center = np.mean(positions, axis=0)

        for position in positions:
            length = np.sum(np.power(center - position, 2))
            steps = ceil(length)
            remainder = abs(length)

            for x, y, z in np.transpose([
                np.linspace(center[0], position[0], steps, dtype=int),
                np.linspace(center[1], position[1], steps, dtype=int),
                np.linspace(center[2], position[2], steps, dtype=int)
            ]):
                if remainder < 0:
                    break

                new_value = max(0.0, min(hematocrit[y][x][z] + remainder, 1.0))
                hematocrit[y][x][z] = new_value
                remainder -= 1.0

    # Set fluid cells outside and on the border of the container to nan, so they can be excluded
    # from further calculations
    hematocrit[np.where(boundary_map == 1)] = np.nan

    return hematocrit
