import sys

import numpy as np
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Annotated
from math import sqrt

import params
from src.linalg import Vector3
from src.sql.entity import Entity, parent

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class CSVCell(Entity):
    iteration_id: Annotated[int, parent('csv_iteration', 'id')]
    prefix: str
    position: Vector3
    area: float
    volume: float
    atomic_block: int
    cell_id: int
    velocity: Vector3


class CSVCellData(Dict[int, Any]):
    def __init__(self, result: List[Tuple[int, any]]):
        super().__init__()

        for cell_id, *values in result:
            if isinstance(values, list):
                values = np.array(values)

            self[cell_id] = values

    @property
    def x(self) -> Dict[int, float]:
        return {cell_id: v[0] for cell_id, v in self.items()}

    @property
    def y(self) -> Dict[int, float]:
        return {cell_id: v[1] for cell_id, v in self.items()}

    @property
    def z(self) -> Dict[int, float]:
        return {cell_id: v[2] for cell_id, v in self.items()}

    @property
    def magnitude(self) -> Dict[int, float]:
        return {cell_id: np.sqrt(np.sum(np.power(np.array(v), 2))) for cell_id, v in self.items()}

    def distance_from_center(self, width: float) -> Dict[int, float]:
        return {cell_id: sqrt(((v[0] - width / 2) ** 2) + ((v[2] - width / 2) ** 2)) for cell_id, v, in self.items()}

    def scale(self, scale):
        for key, value in self.items():
            self[key] = value * scale

        return self


@dataclass
class CSVCells:
    connection: Any
    iteration_id: int
    prefix: str

    def _get_value(self, column: str or None = None) -> CSVCellData:
        if column is None:
            # The name of the parent function is the column it should query
            column = inspect.stack()[1].function

        result = self.connection.select_all(
            f"SELECT cell_id, {column} FROM csv_cell WHERE iteration_id=? AND prefix=?;",
            self.iteration_id,
            self.prefix
        )

        return CSVCellData(result)
    
    @property
    def position(self) -> CSVCellData:
        return self._get_value('position_x, position_y, position_z')

    @property
    def velocity(self) -> CSVCellData:
        return self._get_value('velocity_x, velocity_y, velocity_z')

    @property
    def atomic_block(self) -> Dict[int, int]:
        return self._get_value()

    @property
    def area(self) -> Dict[int, float]:
        return self._get_value()

    @property
    def volume(self) -> Dict[int, float]:
        return self._get_value()


def create_csv_cells(connection: "Connection", iteration_id: int, file: Path, prefix: str):
    with open(file) as f:
        columns = {column: i for i, column in enumerate(f.readline().rstrip().split(','))}

        cells = []
        for i, line in enumerate(f.readlines()):
            values = line.rstrip().split(',')

            data = {}
            for hdf5_name, entity_name in params.CSV_CELL_FIELDS:
                try:
                    data[entity_name] = values[columns[hdf5_name]]
                except KeyError:
                    print(f"Could not find the '{hdf5_name}' column in the {prefix} cell CSV dataset", file=sys.stderr)

            try:
                cells.append(
                    CSVCell.from_dict(
                        iteration_id=iteration_id,
                        prefix=prefix,
                        **data
                    )
                )
            except Exception as e:
                print(f"Could not parse CSV {prefix} data at iteration {iteration_id} at line {i} with the values: {values}", file=sys.stderr)
                raise e

        CSVCell.insert_many(connection, cells)
