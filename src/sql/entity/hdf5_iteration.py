from src.sql.entity import Entity, parent, exclude
from src.sql.entity.hdf5_cell import Hdf5Cells
from src.sql.entity.hdf5_fluid import HDF5Fluid

from dataclasses import dataclass
from typing import Dict, Any, Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Hdf5Iteration(Entity):
    simulation_id: Annotated[int, parent('simulation', 'id')]
    iteration: int
    connection: Annotated[Any, exclude] = None

    @property
    def fluid(self) -> HDF5Fluid:
        return HDF5Fluid(iteration_id=self.id, connection=self.connection)

    @property
    def red_blood_cells(self) -> Hdf5Cells:
        return Hdf5Cells(self.connection, self.id, "RBC")

    @property
    def platelets(self) -> Hdf5Cells:
        return Hdf5Cells(self.connection, self.id, "PLT")


def load_hdf5_iterations(connection: "Connection", simulation_id: int) -> Dict[int, Hdf5Iteration]:
    return {i.iteration: i for i in Hdf5Iteration.load_all(connection, simulation_id)}
