from collections import defaultdict
from dataclasses import dataclass
from glob import glob
import h5py
import inspect
import numpy as np
from typing import Any, Dict, List, Tuple


@dataclass
class Hdf5Cell:
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
    id: int = None

    def append(self, area_force, bending_force, inner_link_force, link_force, position, repulsion_force, total_force, velocity, viscous_force, volume_force):
        self.area_force = np.append(self.area_force, [area_force], axis=0)
        self.bending_force = np.append(self.bending_force, [bending_force], axis=0)
        self.inner_link_force = np.append(self.inner_link_force, [inner_link_force], axis=0)
        self.link_force = np.append(self.link_force, [link_force], axis=0)
        self.position = np.append(self.position, [position], axis=0)
        self.repulsion_force = np.append(self.repulsion_force, [repulsion_force], axis=0)
        self.total_force = np.append(self.total_force, [total_force], axis=0)
        self.velocity = np.append(self.velocity, [velocity], axis=0)
        self.viscous_force = np.append(self.viscous_force, [viscous_force], axis=0)
        self.volume_force = np.append(self.volume_force, [volume_force], axis=0)

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE hdf5_cell (
                    id integer PRIMARY KEY,
                    prefix text,
                    iteration_id integer,
                    area_force array,
                    bending_force array,
                    inner_link_force array,
                    link_force array,
                    position array,
                    repulsion_force array,
                    total_force array,
                    velocity array,
                    viscous_force array,
                    volume_force array,
                    FOREIGN KEY (iteration_id) REFERENCES hdf5_iteration (id) ON DELETE CASCADE      
                  );"""


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

        return Hdf5CellData(self.connection.select_all(f"""SELECT {column} FROM hdf5_cell WHERE iteration_id='{self.iteration_id}' AND prefix='{self.prefix}';"""))

    @property
    def count(self) -> int:
        return self.connection.select_one(f"SELECT COUNT(*) FROM hdf5_cell WHERE iteration_id='{self.iteration_id}' AND prefix='{self.prefix}'")[0]

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


def create_hdf5_cells(connection, iteration_id: int, directory: str, prefix: str) -> List[np.ndarray]:
    cells: Dict[int, Hdf5Cell] = defaultdict(Hdf5Cell)

    files = sorted([f for f in glob(directory + f"{prefix}.*.p.*.h5")])
    for file in files:
        with h5py.File(file, 'r') as f:
            cell_id = np.array(f['Cell Id'], dtype=int)
            area_force = np.array(f['Area force'])
            bending_force = np.array(f['Bending force'])
            inner_link_force = np.array(f['Inner link force'])
            link_force = np.array(f['Link force'])
            position = np.array(f['Position'])
            repulsion_force = np.array(f['Repulsion force'])
            total_force = np.array(f['Total force'])
            velocity = np.array(f['Velocity'])
            viscous_force = np.array(f['Viscous force'])
            volume_force = np.array(f['Volume force'])

        for i in range(len(cell_id)):
            cells[cell_id[i][0]].append(
                area_force[i],
                bending_force[i],
                inner_link_force[i],
                link_force[i],
                position[i],
                repulsion_force[i],
                total_force[i],
                velocity[i],
                viscous_force[i],
                volume_force[i]
            )

    connection.insert_many(
        f"""
            INSERT INTO hdf5_cell (iteration_id, prefix, area_force, bending_force, inner_link_force, link_force, position, repulsion_force, total_force, velocity, viscous_force, volume_force) 
            VALUES ('{iteration_id}', '{prefix}', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        list(map(
            lambda cell: (cell.area_force, cell.bending_force, cell.inner_link_force, cell.link_force, cell.position, cell.repulsion_force, cell.total_force, cell.velocity, cell.viscous_force, cell.volume_force),
            cells.values()
        ))
    )

    return [cell.position for cell in cells.values()]
