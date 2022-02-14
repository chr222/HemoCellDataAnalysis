import sys
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
import h5py
import inspect
import numpy as np
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Annotated

import params
from src.sql.entity import Entity, parent

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Hdf5Cell(Entity):
    prefix: str
    iteration_id: Annotated[int, parent('hdf5_iteration', 'id')]
    area_force: np.ndarray = np.empty((0, 3))
    bending_force: np.ndarray = np.empty((0, 3))
    inner_link_force: np.ndarray = np.empty((0, 3))
    link_force: np.ndarray = np.empty((0, 3))
    position: np.ndarray = np.empty((0, 3))
    repulsion_force: np.ndarray = np.empty((0, 3))
    total_force: np.ndarray = np.empty((0, 3))
    velocity: np.ndarray = np.empty((0, 3))
    viscous_force: np.ndarray = np.empty((0, 3))
    volume_force: np.ndarray = np.empty((0, 3))

    def __getitem__(self, item):
        return getattr(self, item)


class Hdf5CellData(List[np.ndarray]):
    def __init__(self, result: List[Tuple[np.ndarray]]):
        super().__init__()

        for (column) in result:
            self.append(column)

    @property
    def x(self) -> List[float]:
        return [v[0] for v in self]

    @property
    def y(self) -> List[float]:
        return [v[1] for v in self]

    @property
    def z(self) -> List[float]:
        return [v[2] for v in self]

    @property
    def magnitude(self) -> List[float]:
        return [np.sqrt(np.sum(np.power(v, 2))) for v in self]


@dataclass
class Hdf5Cells:
    connection: Any
    iteration_id: int
    prefix: str

    def _get_value(self, column: str = None) -> Hdf5CellData:
        # The name of the parent function is the column it should query
        if column is None:
            column = inspect.stack()[1].function

        return Hdf5CellData(self.connection.select_all(
            f"SELECT {column} FROM hdf5_cell WHERE iteration_id=? AND prefix=?;",
            self.iteration_id,
            self.prefix
        ))

    @property
    def count(self) -> int:
        return self.connection.select_one(
            f"SELECT COUNT(*) FROM hdf5_cell WHERE iteration_id=? AND prefix=?",
            self.iteration_id,
            self.prefix
        )[0]

    @property
    def area_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def bending_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def inner_link_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def link_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def position(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def repulsion_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def total_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def velocity(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def viscous_force(self) -> Hdf5CellData:
        return self._get_value()

    @property
    def volume_force(self) -> Hdf5CellData:
        return self._get_value()


def create_hdf5_cells(connection: "Connection", iteration_id: int, directory: str, prefix: str) -> List[np.ndarray] or None:
    cells: Dict[int, Hdf5Cell] = defaultdict(lambda: Hdf5Cell(iteration_id=iteration_id, prefix=prefix))

    files = sorted([f for f in glob(directory + f"{prefix}.*.p.*.h5")])
    for file in files:
        with h5py.File(file, 'r') as f:
            try:
                cell_id = np.array(f['Cell Id'], dtype=int)
            except KeyError:
                return None

            for hdf5_name, entity_name in params.HDF5_CELL_FIELDS:
                try:
                    dataset = np.array(f[hdf5_name])
                except KeyError:
                    print(f"Could not find '{hdf5_name}' in the HDF5 {prefix} cell dataset", file=sys.stderr)
                    continue

                for i in range(len(cell_id)):
                    np.append(cells[cell_id[i][0]][entity_name], [dataset[i]], axis=0)

    Hdf5Cell.insert_many(connection, list(cells.values()))
    return [cell.position for cell in cells.values()]
