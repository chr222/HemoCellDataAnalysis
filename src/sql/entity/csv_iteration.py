from src.sql.entity.csv_cell import CSVCells

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class CSVIteration:
    connection: Any
    iteration: int
    id: int = None

    @property
    def red_blood_cells(self) -> CSVCells:
        return CSVCells(self.connection, self.id, "RBC")

    @property
    def platelets(self) -> CSVCells:
        return CSVCells(self.connection, self.id, "PLT")

    def insert(self, connection, simulation_id: int):
        self.id = connection.insert(f"INSERT INTO csv_iteration (simulation_id, iteration) VALUES ('{simulation_id}', '{self.iteration}');")

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE csv_iteration (
                    id integer PRIMARY KEY,
                    simulation_id integer,
                    iteration integer,
                    FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON DELETE CASCADE   
                  );"""


def load_csv_iterations(connection, simulation_id: int) -> Dict[int, CSVIteration]:
    result = connection.select_all(f"SELECT iteration, id FROM csv_iteration WHERE simulation_id='{simulation_id}'")

    return {iteration: CSVIteration(connection, iteration, iteration_id) for iteration, iteration_id in result}
