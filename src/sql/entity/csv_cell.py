import numpy as np
from dataclasses import dataclass
import inspect
from typing import Any, Dict, List, Tuple
from math import sqrt


@dataclass
class CSVCell:
    position: np.ndarray
    area: float
    volume: float
    atomic_block: int
    cell_id: int
    velocity: np.ndarray
    id: int = None

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE csv_cell (
                    id integer PRIMARY KEY,
                    prefix text,
                    iteration_id integer,
                    position_x real,
                    position_y real,
                    position_z real,
                    area real,
                    volume real,
                    atomic_block integer,
                    cell_id integer,
                    base_cell_id integer,
                    velocity_x real,
                    velocity_y real,
                    velocity_z real,
                    FOREIGN KEY (iteration_id) REFERENCES csv_iteration (id) ON DELETE CASCADE    
                  );"""


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

        result = self.connection.select_all(f"SELECT cell_id, {column} FROM csv_cell WHERE iteration_id='{self.iteration_id}' AND prefix='{self.prefix}';")

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


def create_csv_cells(connection, iteration_id: int, file: str, prefix: str):
    with open(file) as f:
        # Skip header
        f.readline()

        cells = []
        for i, line in enumerate(f.readlines()):
            values = line.rstrip().split(',')

            try:
                cells.append(
                    CSVCell(
                        position=np.array([float(values[0]), float(values[1]), float(values[2])]),
                        area=float(values[3]),
                        volume=float(values[4]),
                        atomic_block=int(values[5]),
                        cell_id=int(values[6]),  # values[7] is baseCellId which is the same as cell_id so we skip that
                        velocity=np.array([float(values[8]), float(values[9]), float(values[10])])
                    )
                )
            except Exception as e:
                print(iteration_id, prefix, i, values)
                raise e

        connection.insert_many(
            f"""
            INSERT INTO csv_cell (iteration_id, prefix, position_x, position_y, position_z, area, volume, atomic_block, cell_id, velocity_x, velocity_y, velocity_z) 
            VALUES ('{iteration_id}', '{prefix}', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            list(map(lambda cell: (*cell.position, cell.area, cell.volume, cell.atomic_block, cell.cell_id, *cell.velocity), cells))
        )
