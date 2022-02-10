from src.sql.entity.hdf5_cell import Hdf5Cells
from src.sql.entity.hdf5_fluid import Fluid

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Hdf5Iteration:
    connection: Any
    iteration: int
    id: int = None

    @property
    def fluid(self) -> Fluid:
        return Fluid(self.connection, self.id)

    @property
    def red_blood_cells(self) -> Hdf5Cells:
        return Hdf5Cells(self.connection, self.id, "RBC")

    @property
    def platelets(self) -> Hdf5Cells:
        return Hdf5Cells(self.connection, self.id, "PLT")

    def insert(self, connection, simulation_id: int):
        self.id = connection.insert(f"INSERT INTO hdf5_iteration (simulation_id, iteration) VALUES ('{simulation_id}', '{self.iteration}');")

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE hdf5_iteration (
                    id integer PRIMARY KEY,
                    simulation_id integer,
                    iteration integer,
                    FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON DELETE CASCADE   
                  );"""


def load_hdf5_iterations(connection, simulation_id: int) -> Dict[int, Hdf5Iteration]:
    result = connection.select_all(f"SELECT iteration, id FROM hdf5_iteration WHERE simulation_id='{simulation_id}'")

    return {iteration: Hdf5Iteration(connection, iteration, iteration_id) for iteration, iteration_id in result}
