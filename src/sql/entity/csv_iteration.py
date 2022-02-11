from src.sql.entity import parent, exclude, Entity
from src.sql.entity.csv_cell import CSVCells

from dataclasses import dataclass
from typing import Dict, Annotated, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class CSVIteration(Entity):
    simulation_id: Annotated[int, parent('simulation', 'id')]
    iteration: int
    connection: Annotated[Any, exclude] = None

    @property
    def red_blood_cells(self) -> CSVCells:
        return CSVCells(self.connection, self.id, "RBC")

    @property
    def platelets(self) -> CSVCells:
        return CSVCells(self.connection, self.id, "PLT")


def load_csv_iterations(connection: "Connection", simulation_id: int) -> Dict[int, CSVIteration]:
    return {i.iteration: i for i in CSVIteration.load_all(connection, simulation_id)}
