from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

import numpy as np
import psutil

from src.progress import StatusHandler, ProgressFunction
from src.sql.entity.csv_iteration import CSVIteration
from src.sql.entity.hdf5_iteration import Hdf5Iteration
from src.sql.entity.csv_cell import CSVCellData
from src.sql.entity.hdf5_fluid import Vector3Matrix, Tensor6Matrix, Tensor9Matrix


class QueryWithProgress:
    def __init__(self, connection, status: StatusHandler, query: str, args: iter):
        self.connection = connection
        self.query = query
        self.args = args
        self.prefix = status.prefix
        self.progress_function = ProgressFunction(len(args), self.prefix)

        self.start_memory_usage = psutil.Process().memory_info().rss
        connection.start_progress_tracker(self.update_progress)
        self.execute()
        connection.stop_progress_tracker()

    def execute(self):
        self.result = []
        for arguments in self.args:
            self.result += self.progress_function.run(self.connection.select_prepared, self.query, arguments)

    @property
    def memory_usage(self) -> float:
        memory_usage = psutil.Process().memory_info().rss

        return (memory_usage - self.start_memory_usage) / (1024 * 1024)

    def update_progress(self):
        self.progress_function.description = f"{self.prefix} ({self.memory_usage:.2f}MB)"


@dataclass
class BulkCollector:
    connection: Any
    simulation_id: int
    hdf5_iterations: Dict[int, Hdf5Iteration]
    csv_iterations: Dict[int, CSVIteration]
    status: StatusHandler

    def csv_iteration_ids(self, n: int = None) -> dict:
        ids = {iteration.id: iteration.iteration for iteration in list(self.csv_iterations.values())}

        if n is not None:
            return {key: ids[key] for key in list(ids.keys())[:n]}

        return ids

    def hdf5_iteration_ids(self, n: int = None) -> dict:
        ids = {iteration.id: iteration.iteration for iteration in list(self.hdf5_iterations.values())}

        if n is not None:
            return {key: ids[key] for key in list(ids.keys())[:n]}

        return ids

    def _get_hdf5_values(self, column: str, limit: int = None) -> list:
        iteration_ids = self.hdf5_iteration_ids(limit)

        query = QueryWithProgress(
            self.connection,
            self.status,
            f"SELECT {column} FROM fluid WHERE iteration_id=?;",
            [(iteration_id,) for iteration_id in iteration_ids.keys()]
        )

        return query.result

    def _get_csv_values(self, prefix: str, column: str, limit: int = None) -> List[CSVCellData]:
        iteration_ids = self.csv_iteration_ids(limit)

        query = QueryWithProgress(
            self.connection,
            self.status,
            f"SELECT iteration_id, cell_id, {column} FROM csv_cell WHERE prefix=? AND iteration_id=?;",
            [(prefix, iteration_id) for iteration_id in iteration_ids.keys()]
        )

        data = {}
        for (iteration_id, cell_id, *column_data) in query.result:
            key = iteration_ids[iteration_id]
            value = (cell_id, *column_data)

            try:
                data[key].append(value)
            except KeyError:
                data[key] = [value]

        return list(map(CSVCellData, list(data.values())))

    def _csv_count(self, prefix: str, limit: int = None) -> List[int]:
        iteration_ids = self.csv_iteration_ids(limit)

        query = QueryWithProgress(
            self.connection,
            self.status,
            f"SELECT COUNT(*) FROM csv_cell WHERE prefix=? AND iteration_id=?",
            [(prefix, iteration_id) for iteration_id in iteration_ids.keys()]
        )

        return [count for (count,) in query.result]

    def fluid_velocities(self, n: int = None) -> List[Vector3Matrix]:
        return [Vector3Matrix(row[0]) for row in self._get_hdf5_values('velocity', n)]

    def fluid_shear_stress(self, n: int = None) -> List[Tensor6Matrix]:
        return [Tensor6Matrix(row[0]) for row in self._get_hdf5_values('shear_stress', n)]

    def fluid_shear_rate(self, n: int = None) -> List[Tensor9Matrix]:
        return [Tensor9Matrix(row[0]) for row in self._get_hdf5_values('shear_rate', n)]

    def platelet_positions(self, n: int = None) -> List[CSVCellData]:
        return self._get_csv_values('PLT', 'position_x, position_y, position_z', n)

    def platelet_velocities(self, n: int = None) -> List[CSVCellData]:
        return self._get_csv_values('PLT', 'velocity_x, velocity_y, velocity_z', n)

    def platelet_counts(self, n: int = None) -> List[int]:
        return self._csv_count('PLT', n)

    def red_blood_cell_positions(self, n: int = None) -> List[CSVCellData]:
        return self._get_csv_values('RBC', 'position_x, position_y, position_z', n)

    def red_blood_cell_velocities(self, n: int = None) -> List[CSVCellData]:
        return self._get_csv_values('RBC', 'velocity_x, velocity_y, velocity_z', n)

    def red_blood_cell_counts(self, n: int = None) -> List[int]:
        return self._csv_count('RBC', n)

    def hematocrits(self, n: int = None) -> List[np.ndarray]:
        return [row for (row,) in self._get_hdf5_values('hematocrit', n)]
